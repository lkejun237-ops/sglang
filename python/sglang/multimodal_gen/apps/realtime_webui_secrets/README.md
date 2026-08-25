# World Studio model-provider secrets

The dual-model launcher uses Doubao by default for Live Direction rewriting,
skill/goal completion, and world-description completion. Put the ignored local
credential at:

`doubao-ark.json`

Its server-only shape is:

```json
{
  "api_key": "...",
  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "model": "doubao-seed-2-0-mini-260428"
}
```

Optional Doubao settings:

- `DOUBAO_API_KEY`
- `DOUBAO_ARK_BASE_URL`
- `DOUBAO_ARK_CREDENTIALS`
- `DOUBAO_MODEL`
- `DOUBAO_REQUEST_TIMEOUT_SECONDS`
- `VIDEO_PROMPT_REWRITE_PROVIDER=doubao`
- `CREATE_WORLD_DESCRIPTION_PROVIDER=doubao`

The secret directory is outside the WebUI static-file root. JSON files here are
ignored by Git and are never served to the browser.

## Gemini fallback

The server reads a Vertex service-account credential from this project-local
path by default:

`prompt-rewriter-vertex.json`

This directory is deliberately outside the WebUI static-file root. The JSON
credential is ignored by Git and is never served to the browser. In a
deployment, mount the Kubernetes Secret at the same path, or set
`VIDEO_PROMPT_REWRITE_CREDENTIALS` to another in-container path.

Optional server-side settings:

- `VIDEO_PROMPT_REWRITE_PROVIDER=vertex`
- `VIDEO_PROMPT_REWRITE_PROJECT_ID`
- `VIDEO_PROMPT_REWRITE_MODEL` (default: `gemini-3.1-flash-lite`)
- `VIDEO_PROMPT_REWRITE_VERTEX_LOCATION` (default: `global`)
- `VIDEO_PROMPT_REWRITE_TIMEOUT_SECONDS` (default: `20`)

World creation also reads `world-image-model-config.json` from this directory.
That ignored JSON contains the project-local `azure/gpt-image-2` model settings.
Deployments may instead set these server-only environment variables:

- `CREATE_WORLD_IMAGE_API_KEY`
- `CREATE_WORLD_IMAGE_ENDPOINT`
- `CREATE_WORLD_IMAGE_API_VERSION`
- `CREATE_WORLD_IMAGE_CONFIG`
- `CREATE_WORLD_CREDENTIALS`
- `CREATE_WORLD_DESCRIPTION_MODEL`
- `CREATE_WORLD_IMAGE_ATTEMPTS` (default: `2`; failed gameplay framing is regenerated)
