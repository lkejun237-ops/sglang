package main

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtcp"
	"github.com/pion/rtp"
)

func TestNativePacerAddsActualDepartureTimeWithoutChangingFrameIdentity(t *testing.T) {
	p := newBoundedPacer(2_000_000)
	defer p.Close()
	adapter := &rembStreamPacer{pacer: p}
	delivered := make(chan rtp.Header, 1)
	writer := adapter.BindLocalStream(&interceptor.StreamInfo{SSRC: 7, RTPHeaderExtensions: []interceptor.RTPHeaderExtension{{URI: absSendTimeURI, ID: 3}}}, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) {
		if len(b) != 300 {
			t.Errorf("payload length changed: %d", len(b))
		}
		delivered <- h.Clone()
		return h.MarshalSize() + len(b), nil
	}))
	before := time.Now()
	_, err := writer.Write(&rtp.Header{Version: 2, SSRC: 7, Timestamp: 90000, SequenceNumber: 42, Marker: true}, make([]byte, 300), nil)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err = p.wait(ctx); err != nil {
		t.Fatal(err)
	}
	h := <-delivered
	if h.Timestamp != 90000 || h.SequenceNumber != 42 || !h.Marker {
		t.Fatal("media identity changed")
	}
	var ast rtp.AbsSendTimeExtension
	if err = ast.Unmarshal(h.GetExtension(3)); err != nil {
		t.Fatal(err)
	}
	after := time.Now()
	sent := ast.Estimate(after)
	if sent.Before(before.Add(-10*time.Microsecond)) || sent.After(after) {
		t.Fatal("timestamp is not the packet departure time")
	}
}

func TestREMBRequiresValidReceiverFeedbackAndKeepsTheCeiling(t *testing.T) {
	for _, rate := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, ok := receiverBitrate(&rtcp.ReceiverEstimatedMaximumBitrate{Bitrate: rate, SSRCs: []uint32{7}}); ok {
			t.Fatalf("accepted %v", rate)
		}
	}
	if _, ok := receiverBitrate(&rtcp.ReceiverEstimatedMaximumBitrate{Bitrate: 1_000_000}); ok {
		t.Fatal("accepted feedback with no media source")
	}
	for _, tc := range []struct {
		input    float32
		expected int
	}{{1_250_000, 1_250_000}, {10_000_000, 4_200_000}, {50_000, 50_000}} {
		if rate, ok := receiverBitrate(&rtcp.ReceiverEstimatedMaximumBitrate{Bitrate: tc.input, SSRCs: []uint32{7}}); !ok || rate != tc.expected {
			t.Fatalf("rate=%d valid=%v", rate, ok)
		}
	}
}

func TestMissingNativeFeedbackStopsUncontrolledSending(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e := &endpoint{ctx: ctx, cancel: cancel, out: make(chan map[string]any, 4), pacer: newBoundedPacer(4_000_000)}
	defer e.pacer.Close()
	now := time.Now()
	if err := e.checkFeedback(now); err != nil {
		t.Fatal(err)
	}
	if err := e.checkFeedback(now.Add(11 * time.Second)); err == nil {
		t.Fatal("continued without receiver feedback")
	}
	e.acceptREMB(&rtcp.ReceiverEstimatedMaximumBitrate{Bitrate: 2_000_000, SSRCs: []uint32{7}}, now.Add(12*time.Second))
	if err := e.checkFeedback(now.Add(13 * time.Second)); err != nil {
		t.Fatal(err)
	}
	if (<-e.out)["bitrate"] != 2_000_000 {
		t.Fatal("native estimate did not reach encoder feedback")
	}
}
