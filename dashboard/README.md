# Valuation Desk dashboard

The published page lives at
<https://claude.ai/artifact/XFFePLr1ynhKV6wJq8xEA7> as a Claude Dashboard
artifact. `index.html` in this directory is the source of truth for that page;
the data it draws comes from five CSV files uploaded to the same artifact.

## Rebuilding the snapshot

```
python -m scripts.build_desk_snapshot --out <dir> --limit 515 --save-every 25
```

The builder picks its universe from SEC `frames` across four market-cap bands
(mega/large, mid, small, emerging) so the list is not all large caps, then
scores every name with `stocklab.scoring`. It is resumable: re-running with the
same `--out` picks up from `snapshot.json` instead of refetching. It writes:

| file                   | what it holds                                      |
| ---------------------- | -------------------------------------------------- |
| `companies.csv`        | one row per company: scores, plain-language notes, valuation |
| `counts.csv`           | headline counts and the snapshot date              |
| `regime_components.csv`| trend / breadth / credit / volatility / macro      |
| `regime_readings.csv`  | the underlying index, credit and macro measures    |
| `regime_summary.csv`   | backdrop label, composite score, exposure          |

## Republishing

1. Upload each CSV to the artifact as an asset. Each upload returns a
   `/_blob/<id>` URL.
2. Point the five `datasets/<name>` documents in the artifact's database at the
   new blob URLs (`source.url`), pinning `if_version` to the version last read.
   Dataset ids: `companies`, `counts`, `regime_components`,
   `regime_readings`, `regime_summary`.
3. Write this `index.html` to the `files/index.html` document if the page
   itself changed.
4. Read `data/users/me/runs` after the page has been opened to confirm every
   dataset loaded and no drawing reported `no_function`.

Every element with a `data-draw` attribute needs a matching registration in the
page script. Do not use `data-draw` as a plain CSS hook — the runtime treats it
as a drawing and reports `no_function`. Use `data-tip` or another attribute for
elements the drawings only query.

A scheduled task runs the rebuild weekly (Sundays, 18:59 America/Chicago). The
page states its as-of date and says plainly that it does not refresh while open.
