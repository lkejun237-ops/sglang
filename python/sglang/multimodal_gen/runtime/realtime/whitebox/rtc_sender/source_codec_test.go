package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net"
	"os"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/pion/ice/v4"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
)

// These AUs were emitted by the real Room software encoders. Receive through
// ICE/DTLS/SRTP and reconstruct elementary bytes with the matching RTP depayer.
func TestSourceCodecsNegotiateAndCarryElementaryAUOverTCP(t *testing.T) {
	raw, err := os.ReadFile("testdata/source-codec-au.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures struct {
		AccessUnits map[string][]byte `json:"accessUnits"`
	}
	if err = json.Unmarshal(raw, &fixtures); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"h264", "vp9", "av1", "h264-high", "session-h264", "session-vp9", "session-av1", "session-h264-high"} {
		t.Run(name, func(t *testing.T) {
			explicit := strings.HasPrefix(name, "session-")
			name = strings.TrimPrefix(name, "session-")
			sourceProfile := ""
			if name == "h264-high" {
				name = "h264"
				sourceProfile = "42c029"
			}
			t.Setenv("WHITEBOX_SOURCE_H264_PROFILE_LEVEL_ID", sourceProfile)
			t.Setenv("WHITEBOX_LOCAL_TEST", "1")
			t.Setenv("WHITEBOX_SOURCE_CODEC", name)
			l, err := net.Listen("tcp4", "127.0.0.1:0")
			if err != nil {
				t.Fatal(err)
			}
			mux := ice.NewTCPMuxDefault(ice.TCPMuxParams{Listener: l, ReadBufferSize: 8})
			defer mux.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			var selection []sourceCodecSelection
			if explicit {
				t.Setenv("WHITEBOX_SOURCE_CODEC", "h264")
				selection = append(selection, sourceCodecSelection{Codec: name, H264Profile: sourceProfile})
			}
			e, err := newEndpoint(ctx, cancel, make(chan map[string]any, 128), nil, mux, "", l.Addr().(*net.TCPAddr).Port, selection...)
			if err != nil {
				t.Fatal(err)
			}
			defer e.pc.Close()
			defer e.pacer.Close()
			media := &webrtc.MediaEngine{}
			if err = media.RegisterDefaultCodecs(); err != nil {
				t.Fatal(err)
			}
			if err = media.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: absSendTimeURI}, webrtc.RTPCodecTypeVideo); err != nil {
				t.Fatal(err)
			}
			settings := webrtc.SettingEngine{}
			settings.SetIncludeLoopbackCandidate(true)
			settings.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeTCP4})
			client, err := webrtc.NewAPI(webrtc.WithMediaEngine(media), webrtc.WithSettingEngine(settings)).NewPeerConnection(webrtc.Configuration{})
			if err != nil {
				t.Fatal(err)
			}
			defer client.Close()
			for _, codecName := range []string{name, "h264"} {
				transceiver, err := client.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly})
				if err != nil {
					t.Fatal(err)
				}
				codec, _, _ := videoCodec(codecName)
				if codecName == name && sourceProfile != "" {
					codec.SDPFmtpLine = "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=" + sourceProfile
				}
				if err = transceiver.SetCodecPreferences([]webrtc.RTPCodecParameters{{RTPCodecCapability: codec.RTPCodecCapability}}); err != nil {
					t.Fatal(err)
				}
			}
			received := make(chan []byte, 1)
			failures := make(chan error, 1)
			client.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
				expected, _, _ := videoCodec(name)
				if track.Codec().MimeType != expected.MimeType {
					failures <- errors.New("unexpected negotiated codec")
					return
				}
				var out []byte
				h264 := &codecs.H264Packet{}
				vp9 := &codecs.VP9Packet{}
				av1 := &codecs.AV1Depacketizer{}
				for {
					packet, _, err := track.ReadRTP()
					if err != nil {
						failures <- err
						return
					}
					var part []byte
					switch name {
					case "h264":
						part, err = h264.Unmarshal(packet.Payload)
					case "vp9":
						part, err = vp9.Unmarshal(packet.Payload)
					case "av1":
						part, err = av1.Unmarshal(packet.Payload)
					}
					if err != nil {
						failures <- err
						return
					}
					out = append(out, part...)
					if packet.Marker {
						received <- out
						return
					}
				}
			})
			gathered := webrtc.GatheringCompletePromise(client)
			offer, err := client.CreateOffer(nil)
			if err != nil {
				t.Fatal(err)
			}
			if err = client.SetLocalDescription(offer); err != nil {
				t.Fatal(err)
			}
			<-gathered
			if err = e.pc.SetRemoteDescription(*client.LocalDescription()); err != nil {
				t.Fatal(err)
			}
			gathered = webrtc.GatheringCompletePromise(e.pc)
			answer, err := e.pc.CreateAnswer(nil)
			if err != nil {
				t.Fatal(err)
			}
			if err = e.pc.SetLocalDescription(answer); err != nil {
				t.Fatal(err)
			}
			<-gathered
			sdp := e.pc.LocalDescription().SDP
			expected, _, _ := videoCodec(name)
			if sourceProfile != "" {
				expected.SDPFmtpLine = "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=" + sourceProfile
			}
			sections := strings.Split(sdp, "m=video ")
			if len(sections) != 3 || !strings.Contains(sections[1], strings.TrimPrefix(expected.MimeType, "video/")+"/90000") || !strings.Contains(sections[2], "H264/90000") {
				t.Fatalf("codec SDP mismatch: %s", sdp)
			}
			if name != "h264" && strings.Contains(sections[1], "H264/90000") {
				t.Fatal("whitebox negotiated an H264 fallback")
			}
			if !strings.Contains(sections[1], expected.SDPFmtpLine) {
				t.Fatal("missing selected source fmtp")
			}
			if err = client.SetRemoteDescription(*e.pc.LocalDescription()); err != nil {
				t.Fatal(err)
			}
			select {
			case <-e.connected:
			case <-time.After(8 * time.Second):
				t.Fatal("ICE did not connect")
			}
			payload := fixtures.AccessUnits[name]
			if err = e.frame(message{ID: 1, Track: "whitebox", Epoch: "fixture", PTS: 0}, payload); err != nil {
				t.Fatal(err)
			}
			select {
			case got := <-received:
				// AV1 RTP intentionally excludes temporal delimiters; remaining OBUs match.
				want := payload
				if name == "av1" && bytes.HasPrefix(want, []byte{0x12, 0}) {
					want = want[2:]
				}
				if name == "h264" {
					// Annex-B start-code length is not part of the NAL identity.
					startCode := regexp.MustCompile("\\x00\\x00(?:\\x00)?\\x01")
					got = startCode.ReplaceAll(got, []byte{0, 0, 0, 1})
					want = startCode.ReplaceAll(want, []byte{0, 0, 0, 1})
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("elementary payload changed: got %d bytes, want %d", len(got), len(want))
				}
			case err := <-failures:
				t.Fatal(err)
			case <-time.After(3 * time.Second):
				t.Fatal("selected codec RTP did not arrive")
			}
		})
	}
}
