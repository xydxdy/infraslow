# infraslow

Tooling for the Bioserenity sleep-study dataset, split into two cohesive
concerns:

1. **PSG/EDF loading** — a deterministic, alias-aware loader that opens a
   subject's EDF recording via [LunaAPI](https://zzz.bwh.harvard.edu/luna/) and
   maps inconsistent physical channel labels onto a fixed set of canonical
   channel names.