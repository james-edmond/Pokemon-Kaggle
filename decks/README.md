# Worlds-2025-Inspired Decks — Pokémon TCG AI Battle (Kaggle)

Eight tournament-style 60-card decks for the [Pokémon TCG AI Battle](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) engine, each adapting a top-performing 2025 World Championships archetype to this competition's curated card pool.

## How to use
Each `<archetype>/deck.csv` is a submission-ready deck list: **60 lines, one card ID per line** (the exact format the engine's `read_deck_csv()` in `main.py` expects). To run one, copy the chosen `deck.csv` next to your `main.py` in a submission.

## Rules these decks satisfy
- **Exactly 60 cards** (engine-enforced — the simulator rejects any other size).
- **At least one Basic Pokémon** (engine-enforced).
- **≤ 4 copies of any card by name**, Basic Energy unlimited (standard TCG rule; self-enforced).
- **≤ 1 ACE SPEC per deck** (standard TCG rule; self-enforced). Note: in this pool the ACE SPEC list is broader than paper — e.g. Master Ball, Hyper Aroma, Prime Catcher, Precious Trolley are all ACE SPEC — so each deck runs exactly one (Prime Catcher, for gust + pivot).

## How they were validated
Every deck was checked against the actual competition simulator (`cg.dll`) via `ptcg.decks`:
- `is_legal` — the engine accepts the 60-card list and starts a battle.
- `is_playable` — 8 self-play (mirror) games per deck run to completion and are won by both seats via real Prize/knockout conditions (no stalls or deck-size rejections).

All eight pass both checks. `deck.csv` files were additionally re-read from disk and re-validated.

## About the card pool
This is a *curated* Scarlet & Violet–era pool, not the exact Worlds 2025 Standard format. It **omits** several 2023 sets (OBF, 151, PAR, PAF) — so Charizard ex, Pidgeot ex, Gholdengo ex and the SVI Gardevoir ex line don't exist here — while **adding** post-Worlds sets (Mega Evolution, Black Bolt, White Flare, Destined Rivals). It's also the "character-engine" era: no generic Professor's Research / Nest Ball / plain Iono, so consistency runs through Poffin, Ultra Ball, Pokégear and archetype-specific engines (Iono's, Marnie's, Future, etc.). Each deck below notes how it adapts.

## The decks


### 1. Dragapult ex — *spread control*
`dragapult-ex/deck.csv` · **Energy identity:** Fire / Psychic

**Worlds 2025 inspiration.** A perennial Day-2 pillar of the 2025 Worlds Standard metagame — prized for closing games two Prizes at a time through the opponent's board.

**Game plan.** Evolve Dreepy → Drakloak → Dragapult ex (Rare Candy skips the middle) and attack with **Phantom Dive** (200 to the Active, plus 6 damage counters spread onto the Bench). The Dusknoir line and Budew turn that chip damage into knockouts and lock the opponent out of Items.

**Key cards.** Dragapult ex (Phantom Dive), Drakloak (Recon Directive draw), Dusknoir / Dusclops (Cursed Blast snipe), Budew (Item lock), Fezandipiti ex (Flip the Script refuel).

**Pool adaptation.** Real Worlds lists ran Pidgeot ex + Rare Candy for consistency; that line (OBF) isn't in this pool, so consistency comes from Pokégear, Poffin and Drakloak instead. Engine note: Phantom Dive costs {R}{P} here (not the paper {P}{P}), hence the Fire/Psychic base.

**Decklist (16 Pokémon / 32 Trainer / 12 Energy):**
- **Pokémon (16)** — 4× Dreepy (TWM · #119); 2× Drakloak (TWM · #120); 3× Dragapult ex (TWM · #121); 2× Duskull (SFA · #131); 1× Dusclops (SFA · #132); 2× Dusknoir (SFA · #133); 1× Budew (PRE · #235); 1× Fezandipiti ex (SFA · #140)
- **Trainer (32)** — 4× Ultra Ball (SVI · #1121); 4× Buddy-Buddy Poffin (TEF · #1086); 4× Rare Candy (SVI · #1079); 3× Pokégear 3.0 (SVI · #1122); 2× Night Stretcher (SFA · #1097); 2× Switch (SVI · #1123); 3× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224); 1× Lacey (SCR · #1199)
- **Energy (12)** — 8× Basic {P} Energy (SVE · #5); 4× Basic {R} Energy (SVE · #2)


### 2. Raging Bolt ex — *turbo ramp*
`raging-bolt-ex/deck.csv` · **Energy identity:** Fighting / Lightning / Grass

**Worlds 2025 inspiration.** Raging Bolt ex + Ogerpon was one of the most-played Day-2 decks at 2025 Worlds — a consistent, hard-hitting Dragon ramp deck.

**Game plan.** Use Teal Mask Ogerpon ex's **Teal Dance** and Ogerpon's Kagura attacks to pile Basic Energy into play, then fire **Bellowing Thunder**, discarding energy for 70 damage each (easily 210–350). **Burst Roar** doubles as a one-Energy 'draw 6'.

**Key cards.** Raging Bolt ex (Bellowing Thunder / Burst Roar), Teal Mask Ogerpon ex (Teal Dance ramp), Crispin & Rosa's Encouragement (energy accel), Sandy Shocks (Ancient sub-attacker).

**Pool adaptation.** Professor Sada's Vitality (the paper deck's accelerator) is absent from this pool, so acceleration leans on Ogerpon, Crispin and Rosa's Encouragement plus a high (22) Energy count — the deck plays a touch slower but keeps the same ceiling.

**Decklist (11 Pokémon / 27 Trainer / 22 Energy):**
- **Pokémon (11)** — 4× Raging Bolt ex (TEF · #63); 2× Teal Mask Ogerpon ex (TWM · #96); 2× Teal Mask Ogerpon (DRI · #349); 1× Fezandipiti ex (SFA · #140); 1× Squawkabilly (DRI · #478); 1× Sandy Shocks (TEF · #312)
- **Trainer (27)** — 4× Ultra Ball (SVI · #1121); 2× Buddy-Buddy Poffin (TEF · #1086); 2× Pokégear 3.0 (SVI · #1122); 2× Night Stretcher (SFA · #1097); 2× Switch (SVI · #1123); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224); 2× Crispin (SCR · #1198); 2× Rosa's Encouragement (POR · #1240)
- **Energy (22)** — 9× Basic {F} Energy (SVE · #6); 8× Basic {L} Energy (SVE · #4); 5× Basic {G} Energy (SVE · #1)


### 3. Marnie's Grimmsnarl ex — *mono-Dark tank*
`marnies-grimmsnarl-ex/deck.csv` · **Energy identity:** Darkness

**Worlds 2025 inspiration.** A breakout of the 2025 Destined Rivals metagame around Worlds — a 320-HP wall that powers itself up as it evolves.

**Game plan.** Evolve into Marnie's Grimmsnarl ex and trigger **Punk Up** to slam up to 5 Basic {D} Energy onto your board in one motion, then hit for **Shadow Bullet** (180 + 30 to the Bench). Spikemuth Gym keeps finding the pieces; Pecharunt ex and Fezandipiti ex give free switching and draw.

**Key cards.** Marnie's Grimmsnarl ex (Punk Up accel + Shadow Bullet), Spikemuth Gym (search Marnie's Pokémon), Pecharunt ex (Subjugating Chains pivot), Fezandipiti ex (Flip the Script).

**Pool adaptation.** Mono-Darkness and almost entirely in-era (Destined Rivals) — the closest-to-faithful build of the eight.

**Decklist (13 Pokémon / 30 Trainer / 17 Energy):**
- **Pokémon (13)** — 4× Marnie's Impidimp (DRI · #646); 2× Marnie's Morgrem (DRI · #647); 3× Marnie's Grimmsnarl ex (DRI · #648); 2× Fezandipiti ex (SFA · #140); 1× Pecharunt ex (SFA · #141); 1× Squawkabilly (DRI · #478)
- **Trainer (30)** — 4× Ultra Ball (SVI · #1121); 4× Buddy-Buddy Poffin (TEF · #1086); 4× Rare Candy (SVI · #1079); 2× Spikemuth Gym (DRI · #1259); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Night Stretcher (SFA · #1097); 2× Switch (SVI · #1123); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224); 1× Lacey (SCR · #1199)
- **Energy (17)** — 17× Basic {D} Energy (SVE · #7)


### 4. Miraidon ex / Iono's Bellibolt — *turbo aggro*
`miraidon-lightning/deck.csv` · **Energy identity:** Lightning

**Worlds 2025 inspiration.** The Lightning-box seat of the Worlds meta. With this pool's SVI Miraidon lacking Tandem Unit, the deck is rebuilt on the **Iono's Bellibolt ex** engine — itself a 2025 Standard archetype.

**Game plan.** Iono's Bellibolt ex's **Electric Streamer** attaches unlimited Basic {L} Energy from hand each turn; Iono's Kilowattrel refuels your hand. That flood of energy powers Miraidon ex's **Cyber Drive** (220) and Bellibolt's **Thunderous Bolt** (230) as early as turn two.

**Key cards.** Iono's Bellibolt ex (Electric Streamer accel), Iono's Kilowattrel (Flashing Draw), Miraidon ex (Cyber Drive 220), Levincia (recur {L} energy).

**Pool adaptation.** Substitutes the Iono's Bellibolt accel engine for the rotated Tandem-Unit Miraidon and the out-of-pool Iron Hands ex; Miraidon ex stays as the marquee closer.

**Decklist (15 Pokémon / 25 Trainer / 20 Energy):**
- **Pokémon (15)** — 3× Iono’s Tadbulb (JTG · #268); 3× Iono’s Bellibolt ex (JTG · #269); 2× Iono’s Wattrel (JTG · #270); 2× Iono’s Kilowattrel (JTG · #271); 2× Iono’s Voltorb (JTG · #265); 1× Iono’s Electrode (JTG · #266); 2× Miraidon ex (TEF · #313)
- **Trainer (25)** — 4× Ultra Ball (SVI · #1121); 2× Buddy-Buddy Poffin (TEF · #1086); 2× Pokégear 3.0 (SVI · #1122); 2× Levincia (JTG · #1254); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Switch (SVI · #1123); 2× Night Stretcher (SFA · #1097); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224)
- **Energy (20)** — 16× Basic {L} Energy (SVE · #4); 4× Basic {P} Energy (SVE · #5)


### 5. Terapagos ex — *bench-scaling toolbox*
`terapagos-ex/deck.csv` · **Energy identity:** Colorless (rainbow) toolbox

**Worlds 2025 inspiration.** Terapagos ex Toolbox was a defining Colorless deck of the late-2024/2025 Standard format and a Worlds staple.

**Game plan.** Flood the Bench (Fan Call, Poffin, Area Zero Underdepths' 8-slot Bench) and swing with **Unified Beatdown** (30× your Benched Pokémon = up to 240). Noctowl's Jewel Seeker digs for exactly what you need each turn; Bloodmoon Ursaluna ex and Latias ex round out the toolbox.

**Key cards.** Terapagos ex (Unified Beatdown / Crown Opal), Noctowl (Jewel Seeker consistency), Area Zero Underdepths (8-Bench), Fan Rotom (turn-1 Fan Call), Bloodmoon Ursaluna ex.

**Pool adaptation.** Rainbow Energy base supports Crown Opal ({G}{W}{L}); Crispin fixes types. Fully in-era cards.

**Decklist (13 Pokémon / 27 Trainer / 20 Energy):**
- **Pokémon (13)** — 3× Terapagos ex (SCR · #176); 3× Hoothoot (SCR · #172); 2× Noctowl (SCR · #173); 2× Fan Rotom (SCR · #174); 1× Bloodmoon Ursaluna ex (TWM · #44); 1× Latias ex (SSP · #184); 1× Squawkabilly (DRI · #478)
- **Trainer (27)** — 4× Ultra Ball (SVI · #1121); 2× Buddy-Buddy Poffin (TEF · #1086); 2× Pokégear 3.0 (SVI · #1122); 2× Area Zero Underdepths (SCR · #1250); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Switch (SVI · #1123); 2× Night Stretcher (SFA · #1097); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224); 2× Crispin (SCR · #1198)
- **Energy (20)** — 5× Basic {G} Energy (SVE · #1); 5× Basic {W} Energy (SVE · #3); 5× Basic {L} Energy (SVE · #4); 5× Basic {P} Energy (SVE · #5)


### 6. Ceruledge ex — *single-Prize punisher*
`ceruledge-ex/deck.csv` · **Energy identity:** Fire / Psychic / Metal

**Worlds 2025 inspiration.** Ceruledge ex was the aggressive Fire deck of the Surging Sparks-era Standard around Worlds — a Stage-1 attacker that snowballs off its own discard pile.

**Game plan.** Turn-2 Ceruledge ex hits with **Abyssal Flames** (30 + 20 per Energy in your discard) that scales fast, or the **Raging Amethyst** nuke (280, discard all Energy). Hearthflame Mask Ogerpon and Energy Recycler keep the fuel cycling.

**Key cards.** Ceruledge ex (Abyssal Flames / Raging Amethyst), Hearthflame Mask Ogerpon (Fire Kagura accel), Energy Recycler, Fezandipiti ex.

**Pool adaptation.** Raging Amethyst's {R}{P}{M} cost drives the three-type Energy base; early game runs on Fire alone via Abyssal Flames.

**Decklist (13 Pokémon / 27 Trainer / 20 Energy):**
- **Pokémon (13)** — 4× Charcadet (SSP · #204); 3× Ceruledge ex (SSP · #320); 2× Hearthflame Mask Ogerpon (DRI · #358); 2× Fezandipiti ex (SFA · #140); 1× Squawkabilly (DRI · #478); 1× Mimikyu (PFL · #767)
- **Trainer (27)** — 4× Ultra Ball (SVI · #1121); 3× Buddy-Buddy Poffin (TEF · #1086); 2× Pokégear 3.0 (SVI · #1122); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Switch (SVI · #1123); 2× Night Stretcher (SFA · #1097); 1× Energy Recycler (DRI · #1139); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224); 2× Crispin (SCR · #1198)
- **Energy (20)** — 13× Basic {R} Energy (SVE · #2); 4× Basic {P} Energy (SVE · #5); 3× Basic {M} Energy (SVE · #8)


### 7. Iron / Future Box — *all-Basic toolbox*
`iron-future-box/deck.csv` · **Energy identity:** Psychic / Grass (Future)

**Worlds 2025 inspiration.** The 'Future' / Iron toolbox that shared the Worlds Lightning-Psychic space — a fast, all-Basic deck with no evolution lines to brick on.

**Game plan.** Miraidon's **Peak Acceleration** loads Basic Energy onto your Future attackers; Iron Crown ex's **Cobalt Command** buffs them all +20. Answer anything with the right tool: Iron Crown ex (50 to two targets), Iron Leaves ex (Prism Edge 180 + free pivot), Iron Boulder (170).

**Key cards.** Miraidon (Peak Acceleration accel), Iron Crown ex (Cobalt Command buff + Twin Shotels), Iron Leaves ex (Rapid Vernier pivot / Prism Edge), Iron Boulder.

**Pool adaptation.** Iron Hands ex (out of pool) is replaced by Iron Boulder and extra Iron Crown ex; every attacker is a Future Basic, so the deck is exceptionally consistent.

**Decklist (13 Pokémon / 27 Trainer / 20 Energy):**
- **Pokémon (13)** — 3× Miraidon (TEF · #87); 3× Iron Crown ex (TEF · #80); 2× Iron Leaves ex (TEF · #75); 2× Iron Boulder (SCR · #971); 2× Fezandipiti ex (SFA · #140); 1× Squawkabilly (DRI · #478)
- **Trainer (27)** — 4× Ultra Ball (SVI · #1121); 2× Buddy-Buddy Poffin (TEF · #1086); 2× Pokégear 3.0 (SVI · #1122); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Switch (SVI · #1123); 2× Night Stretcher (SFA · #1097); 2× Energy Retrieval (SVI · #1118); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 2× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224); 2× Judge (SVI · #1213)
- **Energy (20)** — 14× Basic {P} Energy (SVE · #5); 6× Basic {G} Energy (SVE · #1)


### 8. Mega Gardevoir ex — *mono-Psychic ramp*
`mega-gardevoir-ex/deck.csv` · **Energy identity:** Psychic

**Worlds 2025 inspiration.** The spiritual successor to the Worlds-mainstay Gardevoir ex ramp deck (the SVI Gardevoir ex line isn't in this pool), rebuilt on the new Mega Evolution card.

**Game plan.** Build Ralts → Kirlia → Mega Gardevoir ex, using **Overflowing Wishes** to fetch a Basic {P} Energy onto every Benched Pokémon, then swing with **Mega Symphonia** (50× total {P} Energy on your Pokémon — scales past 300). Kirlia's Call Sign keeps the hand stocked.

**Key cards.** Mega Gardevoir ex (Overflowing Wishes accel + Mega Symphonia), Kirlia (Call Sign search), Rare Candy, Budew (Item lock).

**Pool adaptation.** A 360-HP mono-Psychic ramp wall; Call Sign + Poffin + Rare Candy support the Stage-2 line in place of the missing Refinement-Kirlia engine.

**Decklist (15 Pokémon / 28 Trainer / 17 Energy):**
- **Pokémon (15)** — 4× Ralts (MEG · #745); 4× Kirlia (MEG · #746); 3× Mega Gardevoir ex (MEG · #747); 1× Budew (PRE · #235); 1× Fezandipiti ex (SFA · #140); 1× Mimikyu (PFL · #767); 1× Squawkabilly (DRI · #478)
- **Trainer (28)** — 4× Ultra Ball (SVI · #1121); 4× Buddy-Buddy Poffin (TEF · #1086); 4× Rare Candy (SVI · #1079); 2× Pokégear 3.0 (SVI · #1122); 2× Boss’s Orders (PAL · #1182); 1× Prime Catcher (TEF · #1088); 2× Switch (SVI · #1123); 2× Night Stretcher (SFA · #1097); 2× Carmine (TWM · #1192); 2× Urbain (ASC · #1236); 1× Lillie's Determination (MEG · #1227); 2× Cheren (WHT · #1224)
- **Energy (17)** — 17× Basic {P} Energy (SVE · #5)


---
*Card IDs reference `pokemon-tcg-ai-battle/EN_Card_Data.csv`. Decks are tuned for engine legality, consistency and a clear game plan; they are starting points, not a solved metagame — fine-tune counts to taste.*
