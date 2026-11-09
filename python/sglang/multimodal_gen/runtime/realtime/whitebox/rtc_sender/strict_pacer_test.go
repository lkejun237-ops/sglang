package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp"
)

// Exercise the actual final writer, rather than a limiter in isolation. An idle
// Pion queue and a bitrate drop must not turn a few bytes of credit into an MTU
// on every 5 ms tick, or bank the whole idle interval as a departure burst.
func TestPacerHonorsLowReceiverRateAfterIdle(t *testing.T) {
	p := newBoundedPacer(4_000_000)
	defer p.Close()
	var sent atomic.Int64
	p.AddStream(77, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) {
		n := h.MarshalSize() + len(b)
		sent.Add(int64(n))
		return n, nil
	}))
	time.Sleep(300 * time.Millisecond)
	p.SetTargetBitrate(50_000)
	start := time.Now()
	for i := 0; i < 20; i++ {
		_, err := p.Write(&rtp.Header{Version: 2, SSRC: 77, SequenceNumber: uint16(i)}, make([]byte, 1200), nil)
		if err != nil {
			t.Fatal(err)
		}
	}
	time.Sleep(250 * time.Millisecond)
	elapsed := time.Since(start)
	// One MTU of burst allowance plus raw receiver feedback, with a small
	// scheduling tolerance. This deliberately excludes a 1.5x rate multiplier.
	allowed := int64(1500 + 50_000*elapsed.Seconds()/8 + 100)
	if got := sent.Load(); got > allowed {
		t.Fatalf("sent %d RTP bytes in %s at 50000 bps; allowed %d", got, elapsed, allowed)
	}
	if sent.Load() == 0 {
		t.Fatal("no packet departed")
	}
}

func TestPacerCloseCancelsPendingLowRateDeparture(t *testing.T) {
	p := newBoundedPacer(1)
	delivered := make(chan struct{}, 10)
	p.AddStream(88, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) {
		delivered <- struct{}{}
		return h.MarshalSize() + len(b), nil
	}))
	// Allow native Pion's first tick to hold credit while the final departure
	// gate is limited to one bit per second.
	p.native.SetTargetBitrate(4_000_000)
	for i := 0; i < 3; i++ {
		if _, err := p.Write(&rtp.Header{Version: 2, SSRC: 88, SequenceNumber: uint16(i)}, make([]byte, 1200), nil); err != nil {
			t.Fatal(err)
		}
	}
	select {
	case <-delivered:
	case <-time.After(time.Second):
		t.Fatal("first packet did not depart")
	}
	time.Sleep(20 * time.Millisecond)
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-delivered:
		t.Fatal("low rate packet escaped before close")
	case <-time.After(30 * time.Millisecond):
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if err := p.wait(ctx); err == nil {
		t.Fatal("closed pacer accepted a drain wait")
	}
}
