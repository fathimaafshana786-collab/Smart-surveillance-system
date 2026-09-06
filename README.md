# Smart Surveillance Video Analytics System

A computer vision project I built to understand how a real surveillance/video-analytics pipeline actually works end to end — not just running a YOLO demo, but detection, tracking, zone/line-based event rules, performance measurement, and a working dashboard, all wired together.

## What this does

Give it a video (a file or a webcam feed) and it will:
- Detect people and vehicles in each frame
- Track them across frames so each object keeps a consistent ID
- Watch for specific events — someone entering a marked restricted zone, crossing a line, loitering too long in one spot, or too many people showing up at once
- Show all of this live in a dashboard, along with real performance numbers (not made-up ones) and a log of every event you can download as a CSV

## Why I built it this way

Most CV portfolio projects I found online are just "load YOLO, draw boxes, done." I wanted something closer to what an actual video analytics system looks like — so I added tracking (not just detection), rule-based events with real temporal logic, config files instead of hardcoded values, logging, and unit tests for the parts of the code that aren't the ML model itself.

This isn't solving a new problem — object detection and tracking are well-established, and commercial products (Axis, Avigilon, etc.) already do this kind of thing. The point was to actually build and understand a full pipeline myself, including the parts that are easy to skip.

## How it's organized

\`\`\`
smart-surveillance/
├── app/
│   ├── detection/       -> YOLO wrapper
│   ├── tracking/        -> ByteTrack wrapper, gives objects persistent IDs
│   ├── analytics/       -> ROI/line-crossing math + per-track position history
│   ├── events/          -> the actual rules (zone entry, line crossing, loitering, count)
│   ├── video/           -> handles file / webcam / RTSP input
│   └── utils/           -> config loading, logging, performance timing
├── config/
│   └── config.yaml      -> everything tunable lives here, not hardcoded in code
├── data/
│   ├── input/            -> put your video here
│   └── output/           -> logs + event CSV get written here
├── tests/                 -> pytest tests for the non-model logic
├── app.py                 -> plain OpenCV window version
├── dashboard.py            -> Streamlit dashboard (the main way to run this)
└── requirements.txt
\`\`\`

## Setting it up

```bash
git clone <your-repo-url>
cd smart-surveillance
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

The YOLO weights (`yolov8n.pt`) download automatically the first time you run it — no manual step needed there.

## Running it

The dashboard is the main way to use this:
```bash
streamlit run dashboard.py
```
It'll open in your browser — check "Start / Resume processing" in the sidebar to begin.

There's also a simpler OpenCV window version, mainly useful for quick debugging:
```bash
python app.py
```
Press `q` to quit that one.

Everything — video source, confidence threshold, ROI coordinates, line position, loitering duration, person-count threshold — is set in `config/config.yaml`, not buried in the code.

## The pipeline, briefly

1. Grab a frame (from file/webcam)
2. Resize it to a consistent size before feeding it to the model
3. Run YOLOv8 detection — filter out low-confidence guesses, and let NMS clean up duplicate/overlapping boxes
4. Run ByteTrack on top of that so each object keeps the same ID across frames
5. Store each track's position history so we can compare "now" against "a moment ago"
6. Check that history against the event rules (did it cross the line? is it in the restricted zone? has it been there too long?)
7. Draw everything and show it in the dashboard, log any events to CSV

## Performance numbers

These are actually measured with `time.perf_counter()` at each stage of the pipeline, not guessed. The dashboard shows inference time, total end-to-end latency, derived FPS, and CPU/memory usage while it's running. Worth noting — inference time (just the model) and end-to-end latency (everything: capture, resize, tracking, event logic, drawing) are genuinely different numbers, and your real FPS is bounded by the latter.

Numbers will vary depending on whatever machine this runs on, so I'm not listing specific figures here — run it yourself and check the dashboard.

## Testing

```bash
pytest tests/ -v
```

22 tests covering the ROI/line-crossing math, the event rules, and config loading. I didn't try to test the YOLO model itself — that's probabilistic, not something you assert exact outputs against — but everything around it (the actual decision logic) is deterministic and testable, so that's what's covered.

While building this, the tests actually caught a couple of real bugs — one where boundary jitter caused the same event to fire multiple times in a row, and one genuine off-by-one bug in how position history was being compared across frames. Both are fixed now.

## Known limitations

- Track IDs can occasionally switch if a person is heavily occluded or crosses paths with someone else — this is a known limitation of tracking-by-detection approaches in general, not something specific to this code
- ROI and line coordinates are placed manually by looking at a sample frame, not auto-calibrated
- Runs on CPU, tuned for a normal laptop rather than high-throughput deployment
- No re-identification — if a track is lost and picked back up as a new ID, the system doesn't know it's the same object

## License

MIT

## Author

Afshana Fathima A