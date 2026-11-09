package main

import "testing"

func TestSessionSourceCodecSelectionIsStrict(t *testing.T) {
	for _, selection := range []sourceCodecSelection{
		{Codec: "vp9"}, {Codec: "av1"}, {Codec: "h264", H264Profile: "42001f"},
	} {
		if err := selection.validate(); err != nil {
			t.Fatal(err)
		}
	}
	for _, selection := range []sourceCodecSelection{
		{Codec: "bogus"}, {Codec: "", H264Profile: "42001f"},
		{Codec: "vp9", H264Profile: "42001f"}, {Codec: "h264", H264Profile: ";extra=1"},
	} {
		if err := selection.validate(); err == nil {
			t.Fatalf("invalid selection accepted: %v", selection)
		}
	}
}
