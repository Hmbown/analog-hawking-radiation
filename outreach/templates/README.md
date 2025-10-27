# Email Templates and Materials

This directory contains all templates for professional outreach to the analogue gravity research community.

## Files

### Email Templates

1. **initial_outreach.txt**
   - Primary email template for first contact
   - Customize for each recipient based on their research area
   - Keep under 200 words
   - Includes customization notes for different researcher types

2. **follow_up.txt**
   - Follow-up email template (after 2+ weeks no response)
   - Multiple variations: new results, check-in, after referral
   - Only send ONE follow-up unless they specifically request updates

3. **collaboration_proposal.txt**
   - Use when researcher expresses interest in collaborating
   - Proposes concrete next steps
   - Includes negotiation tips and handling different scenarios

### Checklists

4. **results_pack_checklist.txt**
   - Complete checklist before sending ANY email
   - Ensures you have concrete results to share
   - Includes PDF creation guide
   - Avoids common mistakes

## Usage Guide

### Before Any Outreach

1. **Run required pipelines** (see pc_cuda_workflow.md):
   - Pipeline A: Baseline run
   - Pipeline B: Universality sweep
   - Pipeline E: κ-inference

2. **Check results_pack_checklist.txt**
   - Verify all required files generated
   - Create 1-page PDF summary
   - Customize template for recipient

3. **Log in outreach_log.csv**
   - Track all outreach attempts
   - Set follow-up reminder

### Template Customization

**For each email:**

1. Replace `[PLACEHOLDER]` text:
   - `[LAST_NAME]` - Their actual last name
   - `[DATE]` - Date of previous email
   - `[SPECIFIC RESULT]` - Concrete results from your runs
   - `[YOUR_NAME]` - Your name
   - `[YOUR_EMAIL]` - Your email

2. Add 1-2 sentences specific to their work:
   - Reference their recent papers
   - Mention specific aspects of their research
   - Connect to your results

3. Verify contact information:
   - Email address from official source
   - Name spelling correct
   - Title (Prof./Dr.) correct

### Email Timing

**Best practices:**
- Tuesday-Thursday, 9am-5pm (their local time)
- Avoid major holidays, conferences, summer vacation
- 2 weeks before initial follow-up
- Maximum 2 total emails (initial + 1 follow-up)

### Attachments

**Required:**
- 1-page PDF summary

**Optional (2-3 max):**
- PNG figures (<1MB each)
- Descriptive filenames

**Never attach:**
- ZIP files
- Large PDFs (>5MB)
- Code or raw data

### Success Metrics

Track in `notes/outreach_log.csv`:
- Total contacted
- Response rate (target: >20%)
- Positive response rate (target: >10%)
- Meeting rate (target: >5%)

## Common Mistakes to Avoid

❌ Sending without running pipelines first
❌ Generic template without customization
❌ Too many attachments
❌ Poor-quality or unlabeled figures
❌ Unverified contact information
❌ Unprofessional tone
❌ Overstating results
❌ Not logging outreach attempts

## Research Categories

Customize based on their research area:

**Water/Fluids:**
- Emphasize: Universality, diagnostic development
- Reference: Water-tank experiments

**BEC:**
- Emphasize: PSD analysis, experimental validation
- Reference: Quantum Hawking radiation observations

**Laser/Plasma:**
- Emphasize: Detection forecasts, PIC integration
- Reference: AnaBHEL, plasma mirrors

**Optical:**
- Emphasize: Negative-frequency radiation, graybody analysis
- Reference: Ultrafast optics, time-varying media

**Theory:**
- Emphasize: Method validation, theoretical assumptions
- Reference: Robustness, dispersive effects

## Professional Standards

✅ Academic professionalism
✅ Respectful tone
✅ Honest about limitations
✅ Offer value, don't just ask
✅ Clear, concise writing
✅ Proper formatting
✅ Timely responses to replies
✅ Gratitude and courtesy

## Getting Help

If you're unsure about:
- Contact information: Check department websites, recent papers
- Template customization: Ask a friend/colleague to review
- Results interpretation: Ask in academic forums (respectfully)

Remember: Professional outreach is about building relationships, not just asking for favors. Provide value first, be respectful, and good things will follow.