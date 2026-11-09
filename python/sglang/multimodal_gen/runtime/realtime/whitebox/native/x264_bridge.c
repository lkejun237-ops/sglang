// SPDX-License-Identifier: Apache-2.0
/* Public x264 API bridge. No x264 private struct access or patched codec. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <x264.h>

typedef struct {
    x264_t *encoder;
    x264_param_t param;
    int64_t last_pts;
    int maxrate;
} wb_encoder;

void *wb_open(int width, int height, int gop, int maxrate) {
    if (width <= 0 || height <= 0 || width % 2 || height % 2 ||
        gop < 1 || gop > 24 || maxrate < 100000 || maxrate > 50000000) return NULL;
    wb_encoder *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    if (x264_param_default_preset(&p->param, "veryfast", "zerolatency") < 0) goto fail;
    p->param.i_width = width;
    p->param.i_height = height;
    p->param.i_csp = X264_CSP_I420;
    p->param.i_threads = 1;
    p->param.i_bframe = 0;
    p->param.b_vfr_input = 0;
    p->param.i_fps_num = 30;
    p->param.i_level_idc = 31;
    p->param.i_fps_den = 1;
    p->param.i_keyint_max = gop;
    p->param.i_scenecut_threshold = 0;
    p->param.b_repeat_headers = 1;
    p->param.b_annexb = 1;
    p->param.rc.i_rc_method = X264_RC_CRF;
    p->param.rc.f_rf_constant = 18;
    p->param.rc.i_vbv_max_bitrate = maxrate / 1000;
    p->param.rc.i_vbv_buffer_size = maxrate / 4000;
    p->param.i_log_level = X264_LOG_ERROR;
    if (x264_param_apply_profile(&p->param, "baseline") < 0) goto fail;
    p->encoder = x264_encoder_open(&p->param);
    if (!p->encoder) goto fail;
    p->last_pts = -1;
    p->maxrate = maxrate;
    return p;
fail:
    x264_param_cleanup(&p->param);
    free(p);
    return NULL;
}

int wb_encode(void *opaque, uint8_t *y, uint8_t *u, uint8_t *v,
              int ys, int us, int vs, int64_t pts, int64_t duration_hint, int force_keyframe,
              uint8_t **out, int *keyframe, int64_t *out_pts) {
    wb_encoder *p = opaque;
    if (!p || !p->encoder || !y || !u || !v || pts < 0 ||
        ys < p->param.i_width || us < p->param.i_width/2 || vs < p->param.i_width/2) return -1;
    int64_t elapsed = p->last_pts < 0 ? 33333 : pts - p->last_pts;
    int64_t delta = duration_hint > 0 ? duration_hint : elapsed;
    if (elapsed <= 0) return -2;
    if (delta <= 0) return -2;
    if (delta > 250000) delta = 250000;
    /* Same bucket capacity, replenished from elapsed source time, never future
       RGB arrival. One fixed encoder tick stands for this elapsed interval. */
    p->param.rc.i_vbv_max_bitrate = (int)((int64_t)p->maxrate * 30 * delta / 1000000000);
    if (p->param.rc.i_vbv_max_bitrate < 1) p->param.rc.i_vbv_max_bitrate = 1;
    if (x264_encoder_reconfig(p->encoder, &p->param) < 0) return -3;
    x264_picture_t input, output;
    x264_picture_init(&input);
    input.img.i_csp = X264_CSP_I420;
    input.img.i_plane = 3;
    input.img.plane[0] = y;
    input.img.plane[1] = u;
    input.img.plane[2] = v;
    input.img.i_stride[0] = ys;
    input.img.i_stride[1] = us;
    input.img.i_stride[2] = vs;
    input.i_pts = pts;
    input.i_type = force_keyframe ? X264_TYPE_IDR : X264_TYPE_AUTO;
    x264_nal_t *nal;
    int count;
    int bytes = x264_encoder_encode(p->encoder, &nal, &count, &input, &output);
    if (bytes <= 0 || count <= 0 || output.i_pts != pts ||
        x264_encoder_delayed_frames(p->encoder) != 0) return -4;
    *out = nal[0].p_payload;
    *keyframe = output.b_keyframe;
    *out_pts = output.i_pts;
    p->last_pts = pts;
    return bytes;
}

void wb_close(void *opaque) {
    wb_encoder *p = opaque;
    if (p) {
        x264_encoder_close(p->encoder);
        x264_param_cleanup(&p->param);
        free(p);
    }
}

int wb_headers(void *opaque, uint8_t **out) {
    wb_encoder *p = opaque;
    if (!p || !p->encoder) return -1;
    x264_nal_t *nal; int count;
    int size = x264_encoder_headers(p->encoder, &nal, &count);
    if (size <= 0 || count <= 0) return -1;
    *out = nal[0].p_payload;
    return size;
}

int wb_bitrate(void *opaque, int maxrate) {
    wb_encoder *p = opaque;
    if (!p || !p->encoder || maxrate < 100000 || maxrate > 50000000) return -1;
    p->maxrate = maxrate;
    p->param.rc.i_vbv_buffer_size = maxrate / 4000;
    return 0;  /* The next picture applies the updated source-time VBV budget. */
}

int wb_abi(void) { return 3; }
const char *wb_version(void) { return X264_POINTVER; }
