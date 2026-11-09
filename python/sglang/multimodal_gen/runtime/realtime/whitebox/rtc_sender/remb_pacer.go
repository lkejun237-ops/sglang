package main

import (
	"errors"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/rtp"
)

const absSendTimeURI = "http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time"

type rembPacerFactory struct{ pacer *boundedPacer }

func (f *rembPacerFactory) NewInterceptor(string) (interceptor.Interceptor, error) {
	return &rembStreamPacer{pacer: f.pacer}, nil
}

type rembStreamPacer struct {
	interceptor.NoOp
	pacer *boundedPacer
}

func (p *rembStreamPacer) BindLocalStream(info *interceptor.StreamInfo, writer interceptor.RTPWriter) interceptor.RTPWriter {
	var astID uint8
	for _, extension := range info.RTPHeaderExtensions {
		if extension.URI == absSendTimeURI {
			astID = uint8(extension.ID)
		}
	}
	if astID == 0 {
		return interceptor.RTPWriterFunc(func(*rtp.Header, []byte, interceptor.Attributes) (int, error) {
			return 0, errors.New("absolute send time was not negotiated")
		})
	}
	p.pacer.AddStream(info.SSRC, interceptor.RTPWriterFunc(func(header *rtp.Header, payload []byte, attrs interceptor.Attributes) (int, error) {
		if astID != 0 {
			value, err := rtp.NewAbsSendTimeExtension(time.Now()).Marshal()
			if err != nil {
				return 0, err
			}
			if err = header.SetExtension(astID, value); err != nil {
				return 0, err
			}
		}
		return writer.Write(header, payload, attrs)
	}))
	return p.pacer
}
func (p *rembStreamPacer) Close() error { return p.pacer.Close() }
