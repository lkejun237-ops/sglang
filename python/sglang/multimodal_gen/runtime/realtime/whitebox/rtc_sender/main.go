// SPDX-License-Identifier: Apache-2.0
// A single attachment's native RTP endpoint. The parent owns the world/lease.
package main

import (
	"bufio"
	"container/list"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/pion/ice/v4"
	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/gcc"
	"github.com/pion/interceptor/pkg/nack"
	"github.com/pion/logging"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
	ratelimit "golang.org/x/time/rate"
)

const maxAU = 1 << 20

// 1 Mbps model + 300 kbps preview, with RTP/control overhead headroom.
// Queue byte/age limits remain independent of this quality floor.
var minTransportBitrate = bitrateSetting("WHITEBOX_RTC_MIN_SEND_BITRATE", 1_500_000)
var maxTransportBitrate = bitrateSetting("WHITEBOX_RTC_MAX_SEND_BITRATE", 4_200_000)

func bitrateSetting(name string, fallback int) int {
	value := os.Getenv(name)
	if value == "" {
		return fallback
	}
	rate, err := strconv.Atoi(value)
	if err != nil || rate < 1 || rate > 100_000_000 {
		panic(name + " must be between 1 and 100000000 bps")
	}
	return rate
}

func init() {
	if minTransportBitrate > maxTransportBitrate {
		panic("WHITEBOX_RTC_MIN_SEND_BITRATE exceeds WHITEBOX_RTC_MAX_SEND_BITRATE")
	}
}

type message struct {
	ID        uint64                   `json:"id,omitempty"`
	Type      string                   `json:"type"`
	SDP       string                   `json:"sdp,omitempty"`
	Candidate *webrtc.ICECandidateInit `json:"candidate,omitempty"`
	Track     string                   `json:"track,omitempty"`
	Epoch     string                   `json:"epoch,omitempty"`
	PTS       int64                    `json:"pts_us,omitempty"`
}

type endpoint struct {
	ctx         context.Context
	cancel      context.CancelFunc
	out         chan map[string]any
	pc          *webrtc.PeerConnection
	pacer       *boundedPacer
	tracks      map[string]*videoTrack
	connected   chan struct{}
	candidates  []*webrtc.ICECandidateInit
	rembMu      sync.Mutex
	firstMedia  time.Time
	lastREMB    time.Time
	lastREMBLog time.Time
}

// This wrapper bounds outstanding writes to Pion's pacer and caps final wire
// departures at the applied transport bitrate. The one-MTU gate prevents Pion's
// per-tick rounding and accumulated idle credit from amplifying congestion.
// Pion still owns the packet queue and retransmission. Up to four AUs overlap;
// each completes on its own actual packet departures.
type boundedPacer struct {
	native     *gcc.LeakyBucketPacer
	departure  *ratelimit.Limiter
	ctx        context.Context
	cancel     context.CancelFunc
	mu         sync.Mutex
	bytes      int
	oldest     time.Time
	queuedAt   *list.List
	change     chan struct{}
	closed     bool
	frames     map[packetIdentity]*frameCompletion
	frameCount int
	once       sync.Once
}

func newBoundedPacer(rate int) *boundedPacer {
	ctx, cancel := context.WithCancel(context.Background())
	return &boundedPacer{native: gcc.NewLeakyBucketPacer(rate), departure: ratelimit.NewLimiter(ratelimit.Limit(rate)/8, 1500), ctx: ctx, cancel: cancel, change: make(chan struct{}), frames: make(map[packetIdentity]*frameCompletion), queuedAt: list.New()}
}
func (p *boundedPacer) notify() { close(p.change); p.change = make(chan struct{}) }
func (p *boundedPacer) AddStream(ssrc uint32, writer interceptor.RTPWriter) {
	p.native.AddStream(ssrc, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, a interceptor.Attributes) (int, error) {
		// Charge RTP plus IPv4/UDP and SRTP authentication overhead. The AST
		// extension is applied by writer.Write, after this wait, so it continues
		// to describe departure rather than enqueue time. Close cancels waits.
		err := p.departure.WaitN(p.ctx, h.MarshalSize()+len(b)+38)
		n := 0
		if err == nil {
			n, err = writer.Write(h, b, a)
		}
		p.mu.Lock()
		p.bytes -= 256 + len(b)
		if p.bytes < 0 {
			p.bytes = 0
		}
		if first := p.queuedAt.Front(); first != nil {
			p.queuedAt.Remove(first)
		}
		p.updateOldest()
		p.finishPacket(h, err)
		p.notify()
		p.mu.Unlock()
		return n, err
	}))
}
func (p *boundedPacer) SetTargetBitrate(rate int) {
	p.departure.SetLimit(ratelimit.Limit(rate) / 8)
	p.native.SetTargetBitrate(rate)
}
func (p *boundedPacer) updateOldest() {
	p.oldest = time.Time{}
	if first := p.queuedAt.Front(); first != nil {
		p.oldest = first.Value.(time.Time)
	}
}
func (p *boundedPacer) Write(h *rtp.Header, b []byte, a interceptor.Attributes) (int, error) {
	size := 256 + len(b)
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed || p.bytes+size > 2<<20 || (!p.oldest.IsZero() && time.Since(p.oldest) > time.Second) {
		return 0, errors.New("native media send budget exceeded")
	}
	queued := p.queuedAt.PushBack(time.Now())
	p.updateOldest()
	p.bytes += size
	// Pion Write only enqueues. Keep our queue in the same order, including RTX.
	n, err := p.native.Write(h, b, a)
	if err != nil {
		p.bytes -= size
		p.queuedAt.Remove(queued)
		p.updateOldest()
		p.notify()
	}
	return n, err
}
func (p *boundedPacer) wait(ctx context.Context) error {
	for {
		p.mu.Lock()
		n, closed, wake := p.bytes, p.closed, p.change
		p.mu.Unlock()
		if closed {
			return errors.New("media sender closed")
		}
		if n == 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-wake:
		}
	}
}
func (p *boundedPacer) Close() error {
	p.once.Do(func() {
		p.cancel()
		p.mu.Lock()
		p.closed = true
		p.cancelFrames()
		p.notify()
		p.mu.Unlock()
		_ = p.native.Close()
	})
	return nil
}

type videoTrack struct {
	track      *webrtc.TrackLocalStaticRTP
	sender     *webrtc.RTPSender
	packetizer rtp.Packetizer
	epoch      string
	firstPTS   int64
	lastPTS    int64
	lastTicks  int64
}

func (e *endpoint) emit(m map[string]any) {
	select {
	case e.out <- m:
	case <-e.ctx.Done():
	default:
		// A blocked parent must not relocate unbounded media/control buffers here.
		e.cancel()
	}
}

// Elementary access units only: VP9 raw frames and AV1 OBU streams, never containers.
func videoCodec(name string) (webrtc.RTPCodecParameters, rtp.Payloader, error) {
	capability := webrtc.RTPCodecCapability{ClockRate: 90000,
		RTCPFeedback: []webrtc.RTCPFeedback{{Type: "nack"}, {Type: "nack", Parameter: "pli"}, {Type: "ccm", Parameter: "fir"}, {Type: "goog-remb"}}}
	var payload webrtc.PayloadType
	var payloader rtp.Payloader
	switch name {
	case "h264":
		capability.MimeType = webrtc.MimeTypeH264
		capability.SDPFmtpLine = "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42c01f"
		payload = 96
		payloader = &codecs.H264Payloader{}
	case "vp9":
		capability.MimeType = webrtc.MimeTypeVP9
		capability.SDPFmtpLine = "profile-id=0"
		payload = 98
		payloader = &codecs.VP9Payloader{}
	case "av1":
		capability.MimeType = webrtc.MimeTypeAV1
		capability.SDPFmtpLine = "profile=0;level-idx=5;tier=0"
		payload = 100
		payloader = &codecs.AV1Payloader{}
	default:
		return webrtc.RTPCodecParameters{}, nil, errors.New("unsupported source codec")
	}
	return webrtc.RTPCodecParameters{RTPCodecCapability: capability, PayloadType: payload}, payloader, nil
}

// An explicit selection is immutable per endpoint; legacy callers pass none.
type sourceCodecSelection struct {
	Codec       string
	H264Profile string
}

func (s sourceCodecSelection) validate() error {
	if _, _, err := videoCodec(s.Codec); err != nil {
		return err
	}
	if s.H264Profile != "" && (s.Codec != "h264" || !regexp.MustCompile(`^[a-f0-9]{6}$`).MatchString(s.H264Profile)) {
		return errors.New("invalid source H264 profile")
	}
	return nil
}

func newEndpoint(ctx context.Context, cancel context.CancelFunc, out chan map[string]any, mux ice.UDPMux, tcpMux ice.TCPMux, publicIP string, publicPort int, selection ...sourceCodecSelection) (*endpoint, error) {
	e := &endpoint{ctx: ctx, cancel: cancel, out: out, tracks: map[string]*videoTrack{}, connected: make(chan struct{})}
	media := &webrtc.MediaEngine{}
	sourceCodec := "h264"
	if os.Getenv("WHITEBOX_LOCAL_TEST") == "1" && os.Getenv("WHITEBOX_SOURCE_CODEC") != "" {
		sourceCodec = os.Getenv("WHITEBOX_SOURCE_CODEC")
	}
	profileOverride := ""
	if len(selection) > 1 {
		return nil, errors.New("invalid source codec selection")
	}
	if len(selection) == 1 {
		if err := selection[0].validate(); err != nil {
			return nil, err
		}
		sourceCodec, profileOverride = selection[0].Codec, selection[0].H264Profile
	}
	selected, _, err := videoCodec(sourceCodec)
	if err != nil {
		return nil, err
	}
	h264, _, _ := videoCodec("h264")
	if sourceCodec == "h264" && (profileOverride != "" || (len(selection) == 0 && os.Getenv("WHITEBOX_LOCAL_TEST") == "1")) {
		profile := profileOverride
		if len(selection) == 0 {
			profile = os.Getenv("WHITEBOX_SOURCE_H264_PROFILE_LEVEL_ID")
		}
		if profile != "" {
			if !regexp.MustCompile(`^[a-f0-9]{6}$`).MatchString(profile) {
				return nil, errors.New("invalid source H264 profile")
			}
			selected.SDPFmtpLine = "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=" + profile
			if selected.SDPFmtpLine != h264.SDPFmtpLine {
				selected.PayloadType = 102
			}
		}
	}
	if err := media.RegisterCodec(h264, webrtc.RTPCodecTypeVideo); err != nil {
		return nil, err
	}
	if sourceCodec != "h264" || selected.SDPFmtpLine != h264.SDPFmtpLine {
		if err := media.RegisterCodec(selected, webrtc.RTPCodecTypeVideo); err != nil {
			return nil, err
		}
	}
	registry := &interceptor.Registry{}
	e.pacer = newBoundedPacer(min(max(4_000_000, minTransportBitrate), maxTransportBitrate))
	logger := logging.NewDefaultLoggerFactory()
	logger.DefaultLogLevel = logging.LogLevelError
	registry.Add(&rembPacerFactory{pacer: e.pacer})
	if err := media.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: absSendTimeURI}, webrtc.RTPCodecTypeVideo); err != nil {
		return nil, err
	}
	if err := webrtc.ConfigureNackWithOptions(media, registry, nil, nack.ResponderSize(256)); err != nil {
		return nil, err
	}
	if err := webrtc.ConfigureRTCPReports(registry); err != nil {
		return nil, err
	}

	settings := webrtc.SettingEngine{LoggerFactory: logger}
	settings.SetLite(true)
	// The isolated browser test binds only loopback. Pion excludes these
	// candidates by default; opt in without changing deployed ICE behavior.
	if os.Getenv("WHITEBOX_LOCAL_TEST") == "1" {
		settings.SetIncludeLoopbackCandidate(true)
	}
	if tcpMux != nil {
		settings.SetICETCPMux(tcpMux)
		settings.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeTCP4})
	} else {
		settings.SetICEUDPMux(mux)
		settings.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeUDP4})
	}
	// Leave five seconds for the outer 45-second negotiation deadline.
	settings.SetICETimeouts(5*time.Second, 40*time.Second, 2*time.Second)
	if publicIP != "" {
		if net.ParseIP(publicIP) == nil {
			return nil, errors.New("invalid public address")
		}
		if err := settings.SetICEAddressRewriteRules(webrtc.ICEAddressRewriteRule{
			External: []string{publicIP}, AsCandidateType: webrtc.ICECandidateTypeHost,
			Mode: webrtc.ICEAddressRewriteReplace,
		}); err != nil {
			return nil, err
		}
	}
	pc, err := webrtc.NewAPI(webrtc.WithMediaEngine(media), webrtc.WithInterceptorRegistry(registry), webrtc.WithSettingEngine(settings)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		return nil, err
	}
	e.pc = pc
	pc.OnICECandidate(func(c *webrtc.ICECandidate) {
		var candidate *webrtc.ICECandidateInit
		if c != nil {
			// Public candidate fields describe the explicit deployment DNAT.
			// ICE Lite replies to valid incoming checks; no private ICE hooks.
			copy := *c
			if publicPort != 0 {
				copy.Port = uint16(publicPort)
			}
			v := copy.ToJSON()
			candidate = &v
		}
		e.emit(map[string]any{"type": "candidate", "candidate": candidate})
	})
	var once sync.Once
	pc.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		e.emit(map[string]any{"type": "state", "state": state.String()})
		if state == webrtc.PeerConnectionStateConnected {
			once.Do(func() { close(e.connected) })
		}
	})
	for _, name := range []string{"whitebox", "model"} {
		codecName := "h264"
		if name == "whitebox" {
			codecName = sourceCodec
		}
		codec, payloader, _ := videoCodec(codecName)
		if name == "whitebox" {
			codec = selected
		}
		track, err := webrtc.NewTrackLocalStaticRTP(codec.RTPCodecCapability, name, "whitebox-rtc")
		if err != nil {
			_ = pc.Close()
			return nil, err
		}
		sender, err := pc.AddTrack(track)
		if err != nil {
			_ = pc.Close()
			return nil, err
		}
		for _, transceiver := range pc.GetTransceivers() {
			if transceiver.Sender() == sender {
				if err := transceiver.SetCodecPreferences([]webrtc.RTPCodecParameters{{RTPCodecCapability: codec.RTPCodecCapability}}); err != nil {
					_ = pc.Close()
					return nil, err
				}
			}
		}
		e.tracks[name] = &videoTrack{track: track, sender: sender, packetizer: rtp.NewPacketizer(1200, uint8(codec.PayloadType), 0, payloader, rtp.NewRandomSequencer(), 90000), lastPTS: -1}
		go func(name string, sender *webrtc.RTPSender) {
			for {
				packets, _, err := sender.ReadRTCP()
				if err != nil {
					return
				}
				for _, packet := range packets {
					switch feedback := packet.(type) {
					case *rtcp.ReceiverEstimatedMaximumBitrate:
						e.acceptREMB(feedback, time.Now())
					case *rtcp.PictureLossIndication, *rtcp.FullIntraRequest:
						e.emit(map[string]any{"type": "keyframe", "track": name})
					}
				}
			}
		}(name, sender)
	}
	return e, nil
}

func (e *endpoint) offer(m message) error {
	if len(m.SDP) == 0 || len(m.SDP) > 128<<10 || e.pc.SignalingState() != webrtc.SignalingStateStable {
		return errors.New("invalid offer")
	}
	if err := e.pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeOffer, SDP: m.SDP}); err != nil {
		return err
	}
	for _, candidate := range e.candidates {
		value := webrtc.ICECandidateInit{}
		if candidate != nil {
			value = *candidate
		}
		if err := e.pc.AddICECandidate(value); err != nil {
			return err
		}
	}
	e.candidates = nil
	answer, err := e.pc.CreateAnswer(nil)
	if err != nil {
		return err
	}
	if err = e.pc.SetLocalDescription(answer); err != nil {
		return err
	}
	// Candidates are sent separately through the public callback with the
	// advertised DNAT port. Never leak an unusable internal port in the SDP.
	lines := []string{}
	for _, line := range strings.Split(e.pc.LocalDescription().SDP, "\r\n") {
		if !strings.HasPrefix(line, "a=candidate:") && line != "a=end-of-candidates" {
			lines = append(lines, line)
		}
	}
	if !strings.Contains(e.pc.LocalDescription().SDP, "goog-remb") || !strings.Contains(e.pc.LocalDescription().SDP, absSendTimeURI) {
		return errors.New("receiver bandwidth feedback is not supported")
	}
	tracks := map[string]string{}
	for _, transceiver := range e.pc.GetTransceivers() {
		if sender := transceiver.Sender(); sender != nil && sender.Track() != nil {
			name := sender.Track().ID()
			if e.tracks[name] != nil && transceiver.Mid() != "" {
				tracks[name] = transceiver.Mid()
			}
		}
	}
	if len(tracks) != 2 {
		return errors.New("video tracks did not negotiate distinct MIDs")
	}
	e.emit(map[string]any{"id": m.ID, "type": "answer", "sdp": strings.Join(lines, "\r\n"), "tracks": tracks})
	return nil
}

func (e *endpoint) frame(m message, data []byte) error {
	t := e.tracks[m.Track]
	if t == nil || m.Epoch == "" || m.PTS < 0 || len(data) == 0 || len(data) > maxAU {
		return errors.New("invalid media")
	}
	select {
	case <-e.connected:
	default:
		return errors.New("media peer is not connected")
	}
	if err := e.checkFeedback(time.Now()); err != nil {
		return err
	}
	var delta int64
	if t.lastPTS < 0 || t.epoch != m.Epoch {
		if t.lastPTS >= 0 {
			delta = 3000
		}
		t.firstPTS, t.lastTicks, t.epoch = m.PTS, 0, m.Epoch
	} else {
		if m.PTS <= t.lastPTS {
			return errors.New("source timestamps must increase")
		}
		ticks := (m.PTS - t.firstPTS) * 90 / 1000
		delta, t.lastTicks = ticks-t.lastTicks, ticks
	}
	t.packetizer.SkipSamples(uint32(delta))
	packets := t.packetizer.Packetize(data, 0)
	if len(packets) == 0 {
		return errors.New("empty packetized frame")
	}
	encodings := t.sender.GetParameters().Encodings
	if len(encodings) != 1 {
		return errors.New("unexpected RTP sender bindings")
	}
	flight, err := e.pacer.beginFrame(uint32(encodings[0].SSRC), packets)
	if err != nil {
		return err
	}
	acceptedAt := time.Now()
	e.emit(map[string]any{"type": "mapped", "id": m.ID, "track": m.Track, "rtp_timestamp": packets[0].Timestamp})
	for _, packet := range packets {
		if err := t.track.WriteRTP(packet); err != nil {
			return err
		}
	}
	t.lastPTS = m.PTS
	// The command loop remains available for the next bounded AU and ICE/control.
	// Reserve timestamp state above before accepting the next frame on this track.
	e.emit(map[string]any{"type": "accepted", "id": m.ID, "track": m.Track, "epoch": m.Epoch})
	go func() {
		timer := time.NewTimer(time.Second)
		defer timer.Stop()
		select {
		case err := <-flight.done:
			if err != nil {
				e.emit(map[string]any{"type": "error", "id": m.ID, "code": "RTC_MEDIA_FAILURE"})
				e.cancel()
				return
			}
			e.emit(map[string]any{"type": "sent", "id": m.ID, "track": m.Track, "rtp_timestamp": packets[0].Timestamp, "packets": len(packets), "pacer_duration_ms": float64(time.Since(acceptedAt).Microseconds()) / 1000})
		case <-timer.C:
			e.emit(map[string]any{"type": "error", "id": m.ID, "code": "RTC_MEDIA_TIMEOUT"})
			e.cancel()
		case <-e.ctx.Done():
		}
	}()
	return nil
}

func readMessage(r io.Reader) (message, []byte, error) {
	var sizes [8]byte
	if _, err := io.ReadFull(r, sizes[:]); err != nil {
		return message{}, nil, err
	}
	n, p := binary.BigEndian.Uint32(sizes[:4]), binary.BigEndian.Uint32(sizes[4:])
	if n == 0 || n > 256<<10 || p > maxAU {
		return message{}, nil, errors.New("IPC budget exceeded")
	}
	raw, data := make([]byte, n), make([]byte, p)
	if _, err := io.ReadFull(r, raw); err != nil {
		return message{}, nil, err
	}
	if _, err := io.ReadFull(r, data); err != nil {
		return message{}, nil, err
	}
	var m message
	if err := json.Unmarshal(raw, &m); err != nil {
		return m, nil, err
	}
	return m, data, nil
}

func run() error {
	listen := flag.String("listen", "0.0.0.0:40000", "Private ICE socket")
	publicIP := flag.String("public-ip", "", "Explicit public IPv4 DNAT address")
	publicPort := flag.Int("public-port", 0, "Explicit public ICE port")
	capabilities := flag.Bool("capabilities", false, "Report compiled elementary RTP codec capabilities without binding sockets")
	sourceCodec := flag.String("source-codec", "", "Explicit codec for this endpoint source only")
	sourceProfile := flag.String("source-h264-profile", "", "Verified source H264 SPS profile-level-id")
	flag.Parse()
	if *capabilities {
		return json.NewEncoder(os.Stdout).Encode(map[string]any{"protocol": "native-video-rtp-v1", "sourceCodecs": []string{"h264", "vp9", "av1"}, "modelCodec": "h264", "sessionSourceEncoding": true})
	}
	var selection []sourceCodecSelection
	if *sourceCodec != "" || *sourceProfile != "" {
		chosen := sourceCodecSelection{Codec: *sourceCodec, H264Profile: *sourceProfile}
		if err := chosen.validate(); err != nil {
			return err
		}
		selection = append(selection, chosen)
	}
	if *publicPort < 0 || *publicPort > 65535 {
		return errors.New("invalid public port")
	}
	network := os.Getenv("WHITEBOX_NATIVE_RTC_NETWORK")
	var mux ice.UDPMux
	var tcpMux ice.TCPMux
	var localPort int
	switch network {
	case "", "udp":
		network = "udp"
		addr, err := net.ResolveUDPAddr("udp4", *listen)
		if err != nil {
			return err
		}
		conn, err := net.ListenUDP("udp4", addr)
		if err != nil {
			return err
		}
		defer conn.Close()
		_ = conn.SetWriteBuffer(64 << 10)
		_ = conn.SetReadBuffer(512 << 10)
		mux = ice.NewUDPMuxDefault(ice.UDPMuxParams{UDPConn: conn})
		defer mux.Close()
		localPort = conn.LocalAddr().(*net.UDPAddr).Port
	case "tcp":
		listener, err := net.Listen("tcp4", *listen)
		if err != nil {
			return err
		}
		tcpMux = ice.NewTCPMuxDefault(ice.TCPMuxParams{Listener: listener, ReadBufferSize: 8, FirstStunBindTimeout: 5 * time.Second})
		defer tcpMux.Close()
		localPort = listener.Addr().(*net.TCPAddr).Port
	default:
		return errors.New("unsupported ICE network: choose udp or tcp")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	out := make(chan map[string]any, 64)
	go func() {
		w := bufio.NewWriter(os.Stdout)
		for {
			select {
			case <-ctx.Done():
				return
			case value := <-out:
				raw, err := json.Marshal(value)
				if err != nil {
					cancel()
					return
				}
				var size [8]byte
				binary.BigEndian.PutUint32(size[:4], uint32(len(raw)))
				if _, err = w.Write(size[:]); err == nil {
					_, err = w.Write(raw)
				}
				if err == nil {
					err = w.Flush()
				}
				if err != nil {
					cancel()
					return
				}
			}
		}
	}()
	e, err := newEndpoint(ctx, cancel, out, mux, tcpMux, *publicIP, *publicPort, selection...)
	if err != nil {
		return err
	}
	defer e.pc.Close()
	ready := map[string]any{"type": "ready", "ice_port": localPort, "network": network, "ice_lite": true}
	ready[network+"_port"] = localPort
	e.emit(ready)
	// Separate bounded input from media pacing: ICE/control can keep progressing.
	incoming := make(chan struct {
		m    message
		data []byte
	}, 1)
	go func() {
		r := bufio.NewReader(os.Stdin)
		for {
			m, data, err := readMessage(r)
			if err != nil {
				cancel()
				return
			}
			select {
			case incoming <- struct {
				m    message
				data []byte
			}{m, data}:
			case <-ctx.Done():
				return
			}
		}
	}()
	for {
		select {
		case <-ctx.Done():
			return nil
		case item := <-incoming:
			m, data := item.m, item.data
			switch m.Type {
			case "offer":
				err = e.offer(m)
			case "candidate":
				if e.pc.RemoteDescription() == nil {
					if len(e.candidates) >= 64 {
						err = errors.New("candidate budget exceeded")
					} else {
						e.candidates = append(e.candidates, m.Candidate)
					}
				} else {
					value := webrtc.ICECandidateInit{}
					if m.Candidate != nil {
						value = *m.Candidate
					}
					err = e.pc.AddICECandidate(value)
				}
				if err == nil {
					e.emit(map[string]any{"id": m.ID, "type": "ok"})
				}
			case "frame":
				err = e.frame(m, data)
			case "close":
				return nil
			default:
				err = errors.New("unsupported media command")
			}
			if err != nil {
				e.emit(map[string]any{"id": m.ID, "type": "error", "code": "RTC_MEDIA_FAILURE"})
				fmt.Fprintln(os.Stderr, "RTC command "+strconv.Quote(m.Type)+":", err)
				return err
			}
		}
	}
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// The browser estimates capacity. This adapter enforces only the deployment
// ceiling and forwards native feedback to the existing pacer and encoders.
func receiverBitrate(feedback *rtcp.ReceiverEstimatedMaximumBitrate) (int, bool) {
	if feedback == nil || len(feedback.SSRCs) == 0 || len(feedback.SSRCs) > 8 {
		return 0, false
	}
	rate := float64(feedback.Bitrate)
	if math.IsNaN(rate) || math.IsInf(rate, 0) || rate < 1 {
		return 0, false
	}
	return int(math.Min(rate, float64(maxTransportBitrate))), true
}
func (e *endpoint) acceptREMB(feedback *rtcp.ReceiverEstimatedMaximumBitrate, now time.Time) {
	rate, ok := receiverBitrate(feedback)
	if !ok {
		return
	}
	e.rembMu.Lock()
	defer e.rembMu.Unlock()
	e.lastREMB = now
	measured := rate
	rate = max(rate, minTransportBitrate)
	e.pacer.SetTargetBitrate(rate)
	e.emit(map[string]any{"type": "bitrate", "bitrate": rate, "receiver_bitrate": measured})
	if now.Sub(e.lastREMBLog) >= time.Second {
		e.lastREMBLog = now
		raw, _ := json.Marshal(map[string]any{"estimator": "receiver_remb", "target_bitrate": rate, "receiver_bitrate": feedback.Bitrate, "ssrc_count": len(feedback.SSRCs), "min_transport_bitrate": minTransportBitrate, "quality_floor_limited": measured < minTransportBitrate})
		fmt.Fprintf(os.Stderr, "WHITEBOX_RTC_CC %s\n", raw)
	}
}
func (e *endpoint) checkFeedback(now time.Time) error {
	e.rembMu.Lock()
	defer e.rembMu.Unlock()
	if e.firstMedia.IsZero() {
		e.firstMedia = now
	}
	last := e.lastREMB
	if last.IsZero() {
		last = e.firstMedia
	}
	if now.Sub(last) > 10*time.Second {
		return errors.New("receiver bandwidth feedback timed out")
	}
	return nil
}
