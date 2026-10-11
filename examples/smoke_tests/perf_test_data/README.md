# Perf-test smoke data

The three files the Gatling simulations read. They are deliberately tiny and
synthetic — enough to prove the load-test path works end to end, not to produce
a meaningful benchmark.

| File | Read by | Notes |
|------|---------|-------|
| `prompts.csv` | `ThrottledTpsCsvSimulation`, `ClosedConcurrencySimulation` | One `prompt` column, 30 rows. The feeder samples at random, so the row count only needs to exceed the concurrency you want variety at. |
| `system-message.txt` | `ChatRequestTemplates` | Becomes the `system` message on every request. |
| `chat-config.json` | `ChatRequestTemplates` | Sampling parameters plus the `model` field sent in the request body. |

## The one field you must keep in sync

`chat-config.json`'s `model` is copied verbatim into each request body, and vLLM
rejects any request whose model name does not match what it is serving. So this
must hold:

```
chat-config.json  "model": "perf-smoke"
       ==
your config       perf_test.serving.served_model_name: perf-smoke
```

Change one and you must change the other, or every request fails with a 404 and
the run reports KO=100%.

## Uploading

The job reads this data from S3, not from the repo, so upload it first and point
`gatling.data_s3_path` at the result:

```bash
AWS_DEFAULT_REGION=us-west-2 aws s3 cp \
  examples/smoke_tests/perf_test_data/ \
  s3://your-bucket/perf_test/smoke_data/ \
  --recursive --exclude README.md
```

Exclude this README — the container copies the whole prefix into the Gatling
resources directory, and a stray Markdown file there is harmless but pointless.

## Using your own prompts

Any CSV with a `prompt` header column works. For production-shaped traffic use
`ThrottledTpsJsonlSimulation` instead, which reads a `prompts.jsonl` of full
chat-completion request bodies and replays them verbatim — see
[`docs/launching-performance-test.md`](../../../docs/launching-performance-test.md).
