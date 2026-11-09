package main

import (
	"context"
	"testing"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
)

type boundContext struct {
	webrtc.TrackLocalContext
	ssrc   webrtc.SSRC
	codec  webrtc.RTPCodecCapability
	writer webrtc.TrackLocalWriter
}

func (b boundContext) CodecParameters() []webrtc.RTPCodecParameters {
	return []webrtc.RTPCodecParameters{{RTPCodecCapability: b.codec, PayloadType: 96}}
}
func (b boundContext) SSRC() webrtc.SSRC                       { return b.ssrc }
func (b boundContext) SSRCRetransmission() webrtc.SSRC         { return 0 }
func (b boundContext) SSRCForwardErrorCorrection() webrtc.SSRC { return 0 }
func (b boundContext) WriteStream() webrtc.TrackLocalWriter    { return b.writer }
func (b boundContext) ID() string                              { return "local-test" }

type boundWriter struct {
	webrtc.TrackLocalWriter
	p *boundedPacer
}

func (w boundWriter) WriteRTP(h *rtp.Header, b []byte) (int, error) { return w.p.Write(h, b, nil) }

// Exercise the actual Pion track's SSRC rewrite and our endpoint command path.
// A local writer replaces the network; there is no cloud/browser/model session.
func TestEndpointAcceptsSuccessorBeforeDepartureAndCompletesWithBoundSSRC(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer pc.Close()
	codec := webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeH264, ClockRate: 90000}
	track, err := webrtc.NewTrackLocalStaticRTP(codec, "model", "test")
	if err != nil {
		t.Fatal(err)
	}
	sender, err := pc.AddTrack(track)
	if err != nil {
		t.Fatal(err)
	}
	ssrc := sender.GetParameters().Encodings[0].SSRC
	p := newBoundedPacer(4_000_000)
	defer p.Close()
	release := make(chan struct{})
	defer cancel()
	p.AddStream(uint32(ssrc), interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) {
		select {
		case <-release:
			return len(b), nil
		case <-ctx.Done():
			return 0, ctx.Err()
		}
	}))
	if _, err = track.Bind(boundContext{ssrc: ssrc, codec: codec, writer: boundWriter{p: p}}); err != nil {
		t.Fatal(err)
	}
	e := &endpoint{ctx: ctx, cancel: cancel, pc: pc, pacer: p, out: make(chan map[string]any, 16), connected: make(chan struct{}), tracks: map[string]*videoTrack{
		"model": {track: track, sender: sender, packetizer: rtp.NewPacketizer(1200, 96, 0, &codecs.H264Payloader{}, rtp.NewRandomSequencer(), 90000), lastPTS: -1},
	}}
	close(e.connected)
	nal := []byte{0, 0, 0, 1, 0x65, 1, 2, 3, 4}
	for i := uint64(1); i <= 2; i++ {
		if err = e.frame(message{ID: i, Track: "model", Epoch: "e", PTS: int64(i-1) * 33333}, nal); err != nil {
			t.Fatal(err)
		}
	}
	mapped := map[uint64]uint32{}
	accepted := 0
	for accepted < 2 {
		select {
		case event := <-e.out:
			switch event["type"] {
			case "mapped":
				mapped[event["id"].(uint64)] = event["rtp_timestamp"].(uint32)
			case "accepted":
				accepted++
			case "sent":
				t.Fatal("reported departure before writer completed")
			}
		case <-time.After(time.Second):
			t.Fatal("successor blocked on departure")
		}
	}
	if mapped[2]-mapped[1] != 2999 {
		t.Fatal("real source PTS was changed")
	}
	close(release)
	seen := map[uint64]bool{}
	for len(seen) < 2 {
		select {
		case event := <-e.out:
			if event["type"] != "sent" {
				t.Fatalf("unexpected completion: %v", event)
			}
			id := event["id"].(uint64)
			if seen[id] || event["rtp_timestamp"] != mapped[id] {
				t.Fatal("completion identity changed")
			}
			seen[id] = true
		case <-time.After(time.Second):
			t.Fatal("bound SSRC did not settle the frame")
		}
	}
}
