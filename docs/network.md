# Network

## Topology

```
┌─────────────────────────────────────────────────────────────────────┐
│  OUTDOORS                                                           │
│                                                                     │
│  ┌──────────────────┐                                               │
│  │   RLC-811A       │                                               │
│  │   4K · IP67      │                                               │
│  │   PoE Camera     │                                               │
│  └────────┬─────────┘                                               │
│           │ Cat6 · 30ft                                             │
│           │ (power + video)                                         │
└───────────┼─────────────────────────────────────────────────────────┘
            │
┌───────────┼─────────────────────────────────────────────────────────┐
│  INDOORS  │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐    ethernet     ┌──────────────────┐          │
│  │   PoE Injector   │ ──────────────► │  Router / Switch │          │
│  │   (bundled w/    │                 │  home network    │          │
│  │    camera)       │                 └────────┬─────────┘          │
│  └────────┬─────────┘                          │ ethernet           │
│           │ 12V AC                             │ (or WiFi)          │
│      Wall outlet                               ▼                    │
│                                      ┌──────────────────┐          │
│                                      │  Raspberry Pi 5  │          │
│                                      │  inference       │          │
│                                      │  GPIO · logs     │          │
│                                      └──────────────────┘          │
│                                                                     │
│  RTSP stream pulled over LAN  →  Pi runs YOLOv8 inference          │
└─────────────────────────────────────────────────────────────────────┘
```

## Connection details

| Link | Medium | Notes |
|---|---|---|
| Camera → PoE injector | Cat6, 30ft outdoor-rated | Carries both 12V power and video data |
| PoE injector → router | Ethernet (LAN port) | Standard home network connection |
| Router → Pi 5 | Ethernet or WiFi | Wired preferred for reliability |
| Pi → camera (RTSP) | LAN (logical) | Pi pulls RTSP stream; camera is just an endpoint on the network |

## RTSP stream

The RLC-811A exposes an RTSP stream accessible on the local network. The Pi connects to this URL and pulls frames for inference — no cloud dependency, no Reolink app required.

Typical Reolink RTSP URL format:
```
rtsp://<username>:<password>@<camera-ip>:554/h264Preview_01_main
```

- Use the main stream for full 4K resolution
- Use the sub stream (`h264Preview_01_sub`) for lower-res / lower-latency if inference throughput is a bottleneck

## Pi network setup

- Configure SSH on first boot (headless setup via `ssh` file on boot partition)
- Assign a static IP or DHCP reservation to the Pi on your router so the camera URL doesn't change
- Same recommendation for the camera's IP

## Security notes

- The RTSP stream is unauthenticated by default on some Reolink firmware versions — set a password in the camera's web UI
- The Pi only needs LAN access; no inbound ports need to be opened on your router
- The FastAPI dashboard (Phase 4) runs on the Pi's LAN IP — only accessible from inside the network unless you explicitly expose it
