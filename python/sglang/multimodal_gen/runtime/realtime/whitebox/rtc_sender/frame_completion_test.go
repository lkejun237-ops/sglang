package main

import (
	"errors"
	"github.com/pion/interceptor"
	"github.com/pion/rtp"
	"sync"
	"testing"
	"time"
)

func packet(ssrc uint32, seq uint16) *rtp.Packet {
	return &rtp.Packet{Header: rtp.Header{Version: 2, SSRC: ssrc, SequenceNumber: seq, Timestamp: 90000, Marker: true}, Payload: make([]byte, 300)}
}

func TestFrameDepartureDoesNotWaitForOtherTrackToDrain(t *testing.T) {
	p := newBoundedPacer(4_000_000)
	defer p.Close()
	release := make(chan struct{})
	defer close(release)
	p.AddStream(1, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) { return len(b), nil }))
	p.AddStream(2, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) { <-release; return len(b), nil }))
	a, b := packet(1, 1), packet(2, 1)
	first, err := p.beginFrame(1, []*rtp.Packet{a})
	if err != nil {
		t.Fatal(err)
	}
	second, err := p.beginFrame(2, []*rtp.Packet{b})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = p.Write(&a.Header, a.Payload, nil); err != nil {
		t.Fatal(err)
	}
	if _, err = p.Write(&b.Header, b.Payload, nil); err != nil {
		t.Fatal(err)
	}
	select {
	case err = <-first.done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("first AU waited for another track")
	}
	select {
	case <-second.done:
		t.Fatal("second AU completed before departure")
	default:
	}
	p.mu.Lock()
	remaining := p.bytes
	p.mu.Unlock()
	if remaining == 0 {
		t.Fatal("test did not retain a pending second AU")
	}
}

func TestFrameCompletionRequiresEveryPacketAndIgnoresRetransmits(t *testing.T) {
	p := newBoundedPacer(4_000_000)
	defer p.Close()
	a, b := packet(3, 1), packet(3, 2)
	f, err := p.beginFrame(3, []*rtp.Packet{a, b})
	if err != nil {
		t.Fatal(err)
	}
	failure := errors.New("socket failed")
	p.mu.Lock()
	p.finishPacket(&b.Header, failure)
	p.finishPacket(&b.Header, nil)
	p.mu.Unlock()
	select {
	case <-f.done:
		t.Fatal("duplicate packet released frame early")
	default:
	}
	p.mu.Lock()
	p.finishPacket(&a.Header, nil)
	p.mu.Unlock()
	if got := <-f.done; got != failure {
		t.Fatalf("write failure lost: %v", got)
	}
	p.mu.Lock()
	p.finishPacket(&a.Header, nil)
	count := p.frameCount
	p.mu.Unlock()
	if count != 0 {
		t.Fatal("frame credit leaked")
	}
}

func TestNativeFrameCreditAndCloseAreBounded(t *testing.T) {
	p := newBoundedPacer(4_000_000)
	var flights []*frameCompletion
	for i := uint16(0); i < 4; i++ {
		f, err := p.beginFrame(5, []*rtp.Packet{packet(5, i)})
		if err != nil {
			t.Fatal(err)
		}
		flights = append(flights, f)
	}
	if _, err := p.beginFrame(5, []*rtp.Packet{packet(5, 4)}); err == nil {
		t.Fatal("accepted fifth AU")
	}
	p.Close()
	for _, f := range flights {
		select {
		case err := <-f.done:
			if err == nil {
				t.Fatal("close reported success")
			}
		case <-time.After(time.Second):
			t.Fatal("close leaked waiter")
		}
	}
	if _, err := p.beginFrame(5, []*rtp.Packet{packet(5, 9)}); err == nil {
		t.Fatal("accepted after close")
	}
}

func TestContinuousQueueAgeTracksRemainingPacketInsteadOfBusyPeriod(t *testing.T) {
	p := newBoundedPacer(4_000_000)
	defer p.Close()
	firstRelease, secondRelease := make(chan struct{}), make(chan struct{})
	var releaseFirst sync.Once
	defer releaseFirst.Do(func() { close(firstRelease) })
	defer close(secondRelease)
	secondEntered := make(chan struct{}, 1)
	p.AddStream(7, interceptor.RTPWriterFunc(func(h *rtp.Header, b []byte, _ interceptor.Attributes) (int, error) {
		if h.SequenceNumber == 1 {
			<-firstRelease
		}
		if h.SequenceNumber == 2 {
			secondEntered <- struct{}{}
			<-secondRelease
		}
		return len(b), nil
	}))
	a, b := packet(7, 1), packet(7, 2)
	if _, err := p.Write(&a.Header, a.Payload, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Write(&b.Header, b.Payload, nil); err != nil {
		t.Fatal(err)
	}
	// Age only the first packet. The second is fresh and will remain pending.
	p.mu.Lock()
	p.queuedAt.Front().Value = time.Now().Add(-2 * time.Second)
	p.updateOldest()
	old := p.oldest
	p.mu.Unlock()
	releaseFirst.Do(func() { close(firstRelease) })
	select {
	case <-secondEntered:
	case <-time.After(time.Second):
		t.Fatal("first packet did not depart")
	}
	p.mu.Lock()
	next, remaining := p.oldest, p.bytes
	p.mu.Unlock()
	if remaining == 0 || !next.After(old) || time.Since(next) > time.Second {
		t.Fatal("queue age still measures the whole busy period")
	}
	c := packet(7, 3)
	if _, err := p.Write(&c.Header, c.Payload, nil); err != nil {
		t.Fatal("fresh successor rejected", err)
	}
}
