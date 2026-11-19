"""Single source of truth for discretization (owner: Member 2). Logic rules rely on these predicate names."""

SPO2_LOW = 90        # spo2 < 90        -> LowOxygen(P)
HR_TACHY = 110       # hr > 110         -> Tachycardia(P)
SBP_LOW = 90         # sbp < 90         -> LowBloodPressure(P)
TEMP_FEVER = 38.5    # temp >= 38.5     -> Fever(P)
AGE_ELDERLY = 65     # age >= 65        -> Elderly(P)
RR_HIGH = 24         # rr >= 24         -> Tachypnea(P)
