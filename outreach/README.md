# Outreach Database

This directory contains our systematic approach to reaching out to the analog Hawking radiation research community. All files in this directory are private (gitignored) to respect privacy and maintain professional networking records.

## Directory Structure

```
outreach/
├── README.md                 # This file
├── contacts/                 # Expert contact database
│   ├── laser_plasma.md      # Laser/plasma & AnaBHEL researchers
│   ├── bec_analogues.md     # BEC analogue black hole researchers
│   ├── water_fluids.md      # Water-waves/fluids researchers
│   ├── optical_analogues.md # Optical analogue researchers
│   ├── theory_foundations.md # Theory & foundations researchers
│   └── priority_list.md     # Priority-ordered contact list
├── templates/                # Email templates and materials
│   ├── initial_outreach.txt # Primary email template
│   ├── follow_up.txt        # Follow-up email template
│   ├── results_pack_checklist.txt # Results packaging checklist
│   └── collaboration_proposal.txt # Collaboration proposal template
├── notes/                    # Interaction tracking
│   ├── outreach_log.csv     # All outreach attempts
│   ├── responses.csv        # Responses received
│   └── meeting_notes/       # Notes from calls/meetings
└── archive/                  # Historical outreach records
    ├── 2024/
    └── 2025/
```

## Workflow

### 1. Before Outreach
- [ ] Run at least 3 pipelines on RTX 3080 (A, B, E from pc_cuda_workflow.md)
- [ ] Collect results: JSON summaries + figures
- [ ] Write 1-page summary PDF
- [ ] Package everything into a "results pack"

### 2. Initial Outreach
1. Check `contacts/priority_list.md` for next contact
2. Open corresponding category file for full profile
3. Use `templates/initial_outreach.txt` as base
4. Customize with specific results
5. Send email
6. Log in `notes/outreach_log.csv`

### 3. Follow-up
- If no response in 2 weeks: send follow-up email (template provided)
- If response: log in `notes/responses.csv`
- Schedule meeting if requested
- Take detailed notes in `notes/meeting_notes/`

## Priority Order

Based on likelihood of engagement and direct relevance:

1. **Silke Weinfurtner** - Water-tank experiments, universality focus
2. **Iacopo Carusotto** - BEC theory, appreciates new tools
3. **Jeff Steinhauer** - Most cited BEC work
4. **Pisin Chen** - AnaBHEL plasma experiments
5. **Daniele Faccio** - Optical analogues
6. **Ralf Schützhold** - Theory foundations
7. **Stefano Liberati** - Phenomenology
8. **Matt Visser** - Conceptual framing

## Best Practices

### Email Strategy
- Lead with concrete figures and JSON results
- Be explicit about limitations
- Offer "apples-to-apples" comparison
- Keep initial email under 200 words
- Attach 1-page PDF summary

### Response Tracking
- Log ALL outreach attempts
- Note date, method, response status
- Track specific requests/feedback received
- Update priority list based on responses

### Professionalism
- Use proper academic titles (Prof., Dr.)
- Respect their time
- Be honest about citizen scientist status
- Offer value, don't just ask for help

## Getting Started

1. Read `contacts/priority_list.md`
2. Pick top 3 contacts from list
3. Run required pipelines (see pc_cuda_workflow.md section 3)
4. Package results
5. Send first outreach email

## Important Notes

- **Privacy**: This directory is gitignored - don't commit personal data
- **Accuracy**: Verify all contact information before sending
- **Timing**: Avoid major conference dates, holidays
- **Follow-through**: If someone expresses interest, follow up promptly

---

*Last updated: 2025-10-26*
