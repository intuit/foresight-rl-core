# Building Docker Training Images

Image builds that install the internal secrets client are part of the Intuit
extension tree and are not in the public package.

Build from the repository root:

```bash
intuit/docker/build.sh --recipe qwen3 --mode vllm --account <account> --repo <repo> --region <region>
```

The Dockerfiles, the build script, and the internal package index are documented
in [`intuit/docs/building-docker-images.md`](../intuit/docs/building-docker-images.md).
