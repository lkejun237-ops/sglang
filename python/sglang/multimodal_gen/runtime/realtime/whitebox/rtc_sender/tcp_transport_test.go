package main

import (
	"context"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/pion/ice/v4"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
)

func TestNativeEndpointConnectsAndDeliversRTPOverICETCP(t *testing.T) {
	l, err := net.Listen("tcp4", "0.0.0.0:0")
	if err != nil {
		t.Fatal(err)
	}
	mux := ice.NewTCPMuxDefault(ice.TCPMuxParams{Listener: l, ReadBufferSize: 8, FirstStunBindTimeout: time.Second})
	defer mux.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e, err := newEndpoint(ctx, cancel, make(chan map[string]any, 64), nil, mux, "", l.Addr().(*net.TCPAddr).Port)
	if err != nil {
		t.Fatal(err)
	}
	defer e.pc.Close()
	defer e.pacer.Close()
	settings := webrtc.SettingEngine{}
	settings.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeTCP4})
	media := &webrtc.MediaEngine{}
	if err := media.RegisterDefaultCodecs(); err != nil {
		t.Fatal(err)
	}
	if err := media.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: absSendTimeURI}, webrtc.RTPCodecTypeVideo); err != nil {
		t.Fatal(err)
	}
	client, err := webrtc.NewAPI(webrtc.WithSettingEngine(settings), webrtc.WithMediaEngine(media)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()
	if _, err = client.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		t.Fatal(err)
	}
	received := make(chan *rtp.Packet, 1)
	client.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		packet, _, readErr := track.ReadRTP()
		if readErr == nil {
			received <- packet
		}
	})
	gotClient := webrtc.GatheringCompletePromise(client)
	offer, err := client.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = client.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gotClient
	if err = e.pc.SetRemoteDescription(*client.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	gotServer := webrtc.GatheringCompletePromise(e.pc)
	answer, err := e.pc.CreateAnswer(nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = e.pc.SetLocalDescription(answer); err != nil {
		t.Fatal(err)
	}
	<-gotServer
	if err = client.SetRemoteDescription(*e.pc.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	select {
	case <-e.connected:
	case <-time.After(8 * time.Second):
		for _, pc := range []*webrtc.PeerConnection{e.pc, client} {
			for _, l := range strings.Split(pc.LocalDescription().SDP, "\n") {
				if strings.HasPrefix(l, "a=candidate:") {
					t.Log(l)
				}
			}
			t.Log("ice", pc.ICEConnectionState(), "connection", pc.ConnectionState())
		}
		t.Fatal("ICE-TCP did not connect")
	}
	want := &rtp.Packet{Header: rtp.Header{Version: 2, SequenceNumber: 17, Timestamp: 90000, Marker: true}, Payload: []byte{0x65, 0x80, 0x42}}
	if err = e.tracks["whitebox"].track.WriteRTP(want); err != nil {
		t.Fatal(err)
	}
	select {
	case got := <-received:
		if got.Timestamp != 90000 || got.SequenceNumber != 17 || len(got.Payload) != 3 {
			t.Fatal("RTP identity changed over TCP")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RTP did not arrive over ICE-TCP")
	}
}
