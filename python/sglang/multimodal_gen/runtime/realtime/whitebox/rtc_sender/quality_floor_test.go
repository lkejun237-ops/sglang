package main

import (
	"context"
	"github.com/pion/rtcp"
	"testing"
	"time"
)

func TestQualityFloorKeepsPacerAndEncoderBudgetAligned(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e := &endpoint{ctx: ctx, cancel: cancel, out: make(chan map[string]any, 8), pacer: newBoundedPacer(4_000_000)}
	defer e.pacer.Close()
	for _, input := range []int{1, 279744, 1000000, 1500000, 2000000, 4200000} {
		e.acceptREMB(&rtcp.ReceiverEstimatedMaximumBitrate{Bitrate: float32(input), SSRCs: []uint32{7, 8}}, time.Now())
		event := <-e.out
		want := max(input, minTransportBitrate)
		if event["bitrate"] != want || event["receiver_bitrate"] != input {
			t.Fatalf("input=%d event=%v expected applied=%d and unchanged receiver", input, event, want)
		}
		if got := int(e.pacer.departure.Limit() * 8); got != want {
			t.Fatalf("encoder budget=%d pacing budget=%d", want, got)
		}
	}
}

func TestShanghaiQualityRange(t *testing.T) {
	oldMin, oldMax := minTransportBitrate, maxTransportBitrate
	defer func() { minTransportBitrate, maxTransportBitrate = oldMin, oldMax }()
	t.Setenv("WHITEBOX_RTC_MIN_SEND_BITRATE", "3200000")
	t.Setenv("WHITEBOX_RTC_MAX_SEND_BITRATE", "5600000")
	minTransportBitrate = bitrateSetting("WHITEBOX_RTC_MIN_SEND_BITRATE", oldMin)
	maxTransportBitrate = bitrateSetting("WHITEBOX_RTC_MAX_SEND_BITRATE", oldMax)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	e := &endpoint{ctx: ctx, cancel: cancel, out: make(chan map[string]any, 8), pacer: newBoundedPacer(4_000_000)}
	defer e.pacer.Close()
	for _, input := range []int{1, 279744, 3200000, 4200000, 5600000, 9000000} {
		e.acceptREMB(&rtcp.ReceiverEstimatedMaximumBitrate{Bitrate: float32(input), SSRCs: []uint32{7, 8}}, time.Now())
		event := <-e.out
		if event["bitrate"] != min(max(input, 3200000), 5600000) {
			t.Fatalf("input=%d event=%v", input, event)
		}
	}
}

func TestBitrateSettingRejectsInvalidValues(t *testing.T) {
	for _, value := range []string{"0", "-1", "not-a-rate", "100000001"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("WHITEBOX_RTC_MIN_SEND_BITRATE", value)
			defer func() {
				if recover() == nil {
					t.Fatal("invalid bitrate accepted")
				}
			}()
			bitrateSetting("WHITEBOX_RTC_MIN_SEND_BITRATE", 1500000)
		})
	}
}
