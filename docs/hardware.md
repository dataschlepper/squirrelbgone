# Hardware

## Components

### Compute
- **Raspberry Pi 5 4GB** — runs inference, controls GPIO, writes logs
  - Powered by a dedicated 27W USB-C supply (e.g. RasTech) with on/off switch
  - During indoor dev: use the official Pi 5 case
  - Outdoors: mount inside the IP65 enclosure using M2.5 brass standoffs

### Camera
- **Reolink RLC-811A** — 4K, 5× optical zoom, IP67, RTSP output
  - Ideal for 15–20 ft detection distance
  - Powered and connected via PoE — single Cat6 cable carries both power and video
  - Comes bundled with a PoE injector

### Relay
- **JBtek 5V 4-channel relay module**
  - Controlled via Pi GPIO signal line
  - **Must be powered by a separate 5V wall adapter** — the Pi's GPIO rail cannot handle the relay's current draw. Shared ground with Pi.
  - Solenoid wires connect to the relay's NO (normally open) and COM terminals

### Solenoid valve
- **Wengart 2W-160-15 12V NC solenoid valve** (direct-acting brass)
  - Normally-closed: valve is shut by default, opens when energized — safe failure mode (no water on power loss)
  - Direct-acting: works from 0 PSI — no minimum water pressure required
  - No polarity on wires
  - Powered by a separate 12V DC wall adapter

### Enclosure
- **IP65 weatherproof ABS project box** — houses the Pi and relay module outdoors
- **IP68 cable glands** (assorted 25-pack) — seal all wire entry points on the enclosure
- **M2.5 brass standoff kit** — mounts Pi inside enclosure

### Plumbing
- **G1/2" to standard hose adapter + misting nozzle**
- **Teflon tape** applied to all solenoid thread connections to prevent leaks

---

## Wiring diagram (text)

```
Wall outlet (12V)
    │
    └──► 12V DC adapter ──► Solenoid valve (one wire)
                                 │
                            Relay COM/NO ◄── Relay module ◄── Pi GPIO pin
                                                    │
                                            Separate 5V adapter
                                            (shared GND with Pi)

Wall outlet (5V)
    └──► 27W USB-C adapter ──► Raspberry Pi 5

Wall outlet (AC)
    └──► PoE injector ──► Cat6 30ft ──► RLC-811A camera
              │
        Router/switch (LAN port)
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Normally-closed solenoid | Valve defaults to shut on power loss — no unintended watering |
| Direct-acting solenoid | Works at 0 PSI; no minimum water pressure required |
| Separate 5V supply for relay | Pi GPIO rail can't handle relay current; sharing causes instability |
| Separate 12V supply for solenoid | Clean isolation from Pi power domain |
| IP65 enclosure + IP68 cable glands | Weatherproofs all electronics for permanent outdoor mounting |
| PoE camera | Single Cat6 cable handles power and video; no separate power run to camera |

---

## Phase 2 dry-run wiring (breadboard)

Before connecting the solenoid, validate GPIO logic with an LED:

1. Wire an LED to a GPIO pin via a 220–330Ω current-limiting resistor on a breadboard
2. Wire detection → GPIO trigger logic in software using `gpiozero`
3. Confirm LED fires on squirrel detection and stays off for birds/other classes
4. Confirm cooldown timer and day/night schedule guard work correctly

Only proceed to relay + solenoid wiring (Phase 3) once the LED dry run is clean.

---

## Phase 3 wiring checklist

- [ ] Relay module: separate 5V supply, shared ground with Pi, GPIO signal line connected
- [ ] NC solenoid wired to relay NO/COM terminals and 12V supply
- [ ] Solenoid connected to hose line with Teflon-taped fittings and misting nozzle
- [ ] Water flow test: confirm valve opens and closes on trigger signal
- [ ] Pi + relay mounted in IP65 enclosure with cable glands sealing all wire entries
