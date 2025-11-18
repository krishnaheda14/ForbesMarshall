# CNC Outsourcing Cost Analysis
**Date**: November 17, 2025

## Problem: Vendors Always More Expensive Than In-House

### Example Operation: J101_Op1
- **Quantity**: 150 units
- **Processing Time**: 94.5 minutes
- **Current Material Cost**: $0.50/unit = $75

---

## Scenario Comparison

### Current State (Why Nothing Changes)

| Hourly Rate | Labor Cost | Material | Total In-House | Vendor Cost | Decision | Savings |
|-------------|------------|----------|----------------|-------------|----------|---------|
| $20/hr | $31.50 | $75 | **$106.50** | $214.65 | In-House | 50% |
| $30/hr | $47.25 | $75 | **$122.25** | $214.65 | In-House | 43% |
| $40/hr | $63.00 | $75 | **$138.00** | $214.65 | In-House | 36% |
| $50/hr | $78.75 | $75 | **$153.75** | $214.65 | In-House | 28% |
| $60/hr | $94.50 | $75 | **$169.50** | $214.65 | In-House | 21% |
| $80/hr | $126.00 | $75 | **$201.00** | $214.65 | In-House | 6% |
| $100/hr | $157.50 | $75 | **$232.50** | $214.65 | **Vendor** | -8% |

**Conclusion**: Need to reach **~$90/hr** before vendors become competitive!

---

## Solution 1: Realistic Vendor Pricing

**Reduce vendor costs by ~60%:**

| Vendor Type | Current Unit Cost | Realistic Cost | Current Transport | Realistic Transport |
|-------------|-------------------|----------------|-------------------|---------------------|
| V_Mill_Std | $0.75 | **$0.35** | $100 | **$40** |
| V_Turn_Std | $0.60 | **$0.25** | $75 | **$30** |
| V_Grind_Std | $0.40 | **$0.18** | $50 | **$25** |
| V_Mill_Pro | $1.50 | **$0.65** | $250 | **$100** |
| V_Turn_Pro | $1.75 | **$0.75** | $200 | **$80** |
| V_Grind_Pro | $1.20 | **$0.55** | $120 | **$50** |

**Result with realistic pricing (J101_Op1):**
- Vendor cost: ($0.35 × 150 + $40) / 0.99 = **$93.43**
- In-house at $30/hr: $122.25
- **Vendor wins by 24%** ✅

---

## Solution 2: Increase Cost Threshold

**Current**: `cost_threshold = 0.7` (outsource only if vendor < 70% of in-house)

**Problem**: Even when vendor is $214 and in-house is $122, ratio is 1.75 > 0.7, so stays in-house!

**Fix**: Change to `cost_threshold = 0.95` or `1.0`

---

## Solution 3: Add Realistic Overhead

**Current in-house cost components:**
- ✅ Labor: hourly_rate × time
- ✅ Material: $0.50/unit
- ❌ Tool wear: ~$0.20/unit
- ❌ Machine depreciation: ~$0.15/unit
- ❌ Utilities: ~$0.10/unit
- ❌ Facility overhead: ~20% of labor

**Enhanced calculation:**
```python
labor_cost = (time / 60) * hourly_rate
material_cost = quantity * 0.50
tooling_cost = quantity * 0.20
overhead = labor_cost * 0.20
total_inhouse = labor_cost + material_cost + tooling_cost + overhead
```

**Result (J101_Op1 at $30/hr):**
- Labor: $47.25
- Material: $75.00
- Tooling: $30.00
- Overhead: $9.45
- **Total: $161.70** (vs vendor $214.65 still cheaper in-house!)

---

## ⚡ RECOMMENDATION

**Best approach**: Combine Solutions 1 + 2

1. **Update vendor_data.csv** with realistic competitive pricing (60% reduction)
2. **Increase cost_threshold** from 0.7 → 0.85
3. **Test at different hourly rates** ($20-$50)

**Expected result:**
- Outsourcing % will vary from 30% (at $20/hr) to 70% (at $50/hr)
- Total cost will increase with hourly rate
- You'll see real trade-offs in the charts!

---

## 🚫 DO NOT DO THIS

**Don't increase hourly rate to $100/hr just to make vendors competitive!**

That would:
- ❌ Make your in-house operations absurdly expensive
- ❌ Not reflect real-world economics
- ❌ Hide the real problem (unrealistic vendor pricing)

---

## Next Steps

Choose ONE approach:

### Quick Fix (2 minutes):
```python
# In sidebar → Outsourcing Policy
# Change cost_threshold from 0.7 → 0.90
```

### Permanent Fix (5 minutes):
1. Edit `data/vendor_data.csv`
2. Reduce unit costs by 50-60%
3. Reduce transport costs by 50-60%
4. Restart app
