package main

import (
	"errors"

	"github.com/pion/rtp"
)

type packetIdentity struct {
	ssrc     uint32
	sequence uint16
}

// Completion means departure from the native pacer, not delivery to the browser.
// Retransmissions do not consume another frame credit.
type frameCompletion struct {
	remaining int
	err       error
	done      chan error
}

func (p *boundedPacer) beginFrame(ssrc uint32, packets []*rtp.Packet) (*frameCompletion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed || p.frameCount >= 4 || len(packets) == 0 {
		return nil, errors.New("native frame credit exceeded")
	}
	for _, packet := range packets {
		if p.frames[packetIdentity{ssrc, packet.SequenceNumber}] != nil {
			return nil, errors.New("native packet identity still in flight")
		}
	}
	f := &frameCompletion{remaining: len(packets), done: make(chan error, 1)}
	for _, packet := range packets {
		p.frames[packetIdentity{ssrc, packet.SequenceNumber}] = f
	}
	p.frameCount++
	return f, nil
}

// Caller holds p.mu. The first actual write settles an original packet once.
func (p *boundedPacer) finishPacket(h *rtp.Header, err error) {
	key := packetIdentity{h.SSRC, h.SequenceNumber}
	f := p.frames[key]
	if f == nil {
		return
	}
	delete(p.frames, key)
	if err != nil && f.err == nil {
		f.err = err
	}
	f.remaining--
	if f.remaining == 0 {
		p.frameCount--
		f.done <- f.err
	}
}

func (p *boundedPacer) cancelFrames() {
	seen := map[*frameCompletion]bool{}
	for _, f := range p.frames {
		if !seen[f] {
			seen[f] = true
			f.done <- errors.New("media sender closed")
		}
	}
	clear(p.frames)
	p.frameCount = 0
}
