# Adaptive mission demo

Start `python -m streamlit run app.py` and open **Adaptive Mission**. The default mode is
**Online satellite monitoring**: no upload, no API key, and no manual download. It queries
[Element 84 Earth Search](https://element84.com/earth-search/) for the 12 latest Sentinel-2 L2A
scenes intersecting the chosen Earth region. A 60-second polling fragment runs while the page
is open, with a 60-second cache per region. Choose Svalbard, Iceland or Alaska. Map tiles and
provider thumbnails also require internet access. This is not a background service after the
browser session closes.

The observer detects new scenes or metadata changes; the quality auditor checks acquisition
age and reported cloud cover; the planner requests image analysis only for scenes no older
than seven days with cloud cover at most 35%. These are explicit quality thresholds, not a
scientific detection rule. Roles rotate across the three policy agents. Repeated unchanged
scene metadata does not generate duplicate handoffs, while changed eligibility triggers review.
Scene IDs, acquisition timestamps, fetch timestamps, response hashes and event parents are
retained in the downloadable evidence. The fetch timestamp is not the observation timestamp.

Connection failure is visible. The page can display its last successful response, labeled as
disconnected, with new tasks paused; it never creates replacement synthetic observations.
The source is an online satellite **catalog** and its map/thumbnail products: it does not
download spectral bands, calculate ice detections, ingest InSAR, or control physical robots.
Earth observations are kept separate from the original lunar science claims.

## Automatic resource seeking

The resource-seeker role reads the live Earth Search `/collections` registry. Collection IDs
are discovered from metadata rather than a fixed secondary-dataset list. It classifies advertised
collections as radar, thermal (including Landsat candidates), or terrain and prioritizes radar
when optical evidence is missing, stale or cloudy. Classification is a discovery hint, not proof
that the advertised modality meets a particular science task.

For each kind, it queries one candidate collection with the watched region's bounding box:
at most one registry request and three item searches per discovery pass, with eight-second
request timeouts and bounded response sizes. Successful discovery passes are cached for ten
minutes per region and evidence-gap priority. It records collection license metadata, item IDs,
asset names, response hashes and query time. Availability means a catalog item intersects the
region; it does not prove freshness, authorization for each asset or sensor suitability.
No full-resolution assets are downloaded during discovery.

New or changed results cause resource seeker → resource auditor → task planner handoffs.
Auditors retain unavailable sources and missing coverage as failures, not measurements.
Planners request source-specific validation for available items. Unchanged results do not produce
duplicate handoffs. Agent roles rotate across these tasks; they remain deterministic policy
agents. The online page displays the resources and includes them in its evidence download.

This is bounded discovery within one trusted public registry, not open-web searching or a
self-installing agent. Catalog text is metadata, not executable instructions. Source-specific
measurement tools, pixel analysis, InSAR processing and hardware adapters remain separate work.
Run `python scripts/resource_discovery_smoke.py` for a real-network discovery check, or
`python -m pytest tests/test_discovery.py tests/test_online.py -v` for offline tests.

Tests: `python -m pytest tests/test_online.py -v` (offline mocked catalog). A real-network check
is available with `python scripts/online_catalog_smoke.py`; it saves catalog and agent evidence
to `/tmp/iceclaw-online-evidence.json`. The test requires internet access and does not spend
model tokens.

For the synthetic robot demo, select **Offline replay** and enable **Auto-play sensor
replay**. No downloaded data or API credentials are needed. Pause and scrub the tick slider
to inspect the map, rover states, current agent roles, and decision log at the same instant.

Three deterministic policy agents lease roles from arriving work: sensor analysis publishes
map updates, evidence audit revisits eligibility, mission planning changes assignments, fleet
recovery releases failed robots' tasks, and sampling records mock collection at arrival.
Each agent can take any of these roles. The rotating lease policy distributes work; it is not
an LLM deciding its own capabilities. Task decisions depend on observations and fleet state.

The example begins with three cold targets and normalized demo optical scores. At tick 3,
a hazard excludes A; tick 4 clears it; tick 5 raises C's priority and redirects a rover.
At tick 7 a rover fails; tick 9 warms B above the analyst-chosen 110 K screening threshold.
The healthy rover continues with remaining eligible targets. These messages are synthetic,
not measurements from a satellite or physical robot. Thresholds and scores do not detect ice.

Download **example sensor input**, edit its readings and upload the JSON array. The adapter
validates message IDs, source labels, source modes, relative ticks, supported measurement kinds,
value ranges and target/robot identifiers. Duplicate IDs and nonfinite values are refused.
Messages are replayed by `(tick, id)`; ticks are simulation time, not UTC acquisition times.
Supported inputs are temperature in kelvin, normalized demo optical index in [0,1], and binary
hazard/robot-fault flags. Targets A/B/C and rover-1/rover-2 belong to a fixed synthetic local grid.
Uploading a `live` label does not authenticate a feed or connect hardware.

Download **full mission evidence and replay** for all observations, events and frame snapshots.
Event parents refer to earlier events, providing a trace from observation through assessment
and action. This is a separate simulation format, not an offline-verified scientific bundle.

Test with `python -m pytest tests/test_mission.py -v`. Tests cover changing observations producing
different decisions, dynamic role leases, valid causal links, task recovery, exclusive assignments,
sampling only after arrival with eligible evidence, invalid inputs, and synchronized UI scrubbing.

Offline replay limits: no physical sensor connection, satellite/InSAR image ingestion, LLM mission planner,
collision avoidance, route hazard checking, freshness policy, energy model or physical assay.
Target hazard flags exclude sampling destinations, not intermediate route cells. The original
four LLM scientific specialists and their verified investigation bundles remain a separate workflow.
To connect physical sensors, an adapter must supply authenticated acquisition timestamps,
georeferenced coordinates, units and quality metadata before extending this fixed-grid prototype.
