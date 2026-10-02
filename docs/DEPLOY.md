# Deploying Semantic Shockwave

The repo's `Dockerfile` builds one container that serves both the API and the
frontend. Anything that runs a Docker image works. The steps below use
**Hugging Face Spaces**, whose free CPU tier (2 vCPU, 16 GB RAM) runs the
embedding model and UMAP comfortably.

## Hugging Face Spaces

1. **Create the Space.** Go to <https://huggingface.co/new-space>, pick
   **Docker** as the SDK, choose the **Blank** template, and leave the hardware on
   *CPU basic* (free).

2. **Give it a Space README.** A Space reads its settings from YAML at the top
   of its own `README.md`. Use this as the Space's README (not this repo's):

   ```markdown
   ---
   title: Semantic Shockwave
   emoji: 💥
   colorFrom: pink
   colorTo: yellow
   sdk: docker
   app_port: 7860
   license: mit
   short_description: Blow away everything but the bridge between two ideas
   ---

   Source: https://github.com/alex-in-CS/semantic-shockwave
   ```

3. **Push the code.** Clone the Space, copy in the project, and push:

   ```bash
   git clone https://huggingface.co/spaces/<you>/semantic-shockwave hf-space
   cd hf-space
   # copy everything from this repo except .git and docs/ (Spaces needs LFS/Xet for binaries)
   cp -r ../semantic-shockwave/{backend,frontend,Dockerfile,.dockerignore,LICENSE} .
   # write the README from step 2, then:
   git add . && git commit -m "Deploy Semantic Shockwave" && git push
   ```

   The Space builds the image (about 5–10 minutes the first time, mostly the
   PyTorch download) and starts it. The first boot also runs UMAP over the
   vocabulary, so allow roughly another minute before the page loads.

4. **Turn on narration (optional).** In the Space's *Settings → Variables and
   secrets*, add a **secret** named `GROQ_API_KEY` with a key from
   <https://console.groq.com/keys>. The Space restarts with narration on. Never
   put the key in the repo.

5. **Link it.** Put the Space URL in this repo's README (replacing the
   "Hosted demo" roadmap item) and in the GitHub repo's *About* box.

### Keeping it in sync (optional)

To redeploy on every push to `main`, add a GitHub Actions job that pushes the
same files to the Space with a Hugging Face write token stored as the repo
secret `HF_TOKEN`. See the Hugging Face guide
[Managing Spaces with GitHub Actions](https://huggingface.co/docs/hub/spaces-github-actions).

## Anywhere else

```bash
docker build -t semantic-shockwave .
docker run -p 7860:7860 -e GROQ_API_KEY=gsk_... semantic-shockwave
```

The platform must route traffic to the container's `PORT` (default 7860; set
`PORT` to change it). The running container uses about 700 MB of RAM, so a
1 GB instance is the practical minimum and 2 GB leaves headroom. It takes
about 90 seconds to boot (UMAP plus JIT compilation) before `/api/v1/health`
answers. To host the frontend
separately (on a static host, say), build it with `VITE_API_URL=https://your-api`
and set `SHOCKWAVE_CORS_ORIGINS` on the backend to the frontend's origin.
