from celestial_fourpoint import (
    FourPointCalculator,
    KUNTZ_SEED_EPSILONS,
    load_kuntz_fixtures,
)


fixture = load_kuntz_fixtures()[0]
calculator = FourPointCalculator.load_precomputed(lmax=15, build_if_missing=True)

print("Kuntz PPPP:", calculator.kuntz_pppp(fixture.vectors))
print("Cached AAAA:", calculator.evaluate(("A", "A", "A", "A"), fixture.vectors, KUNTZ_SEED_EPSILONS))

for entry in calculator.compare(fixture.vectors, KUNTZ_SEED_EPSILONS, backend="auto"):
    print("".join(entry.observables), entry.relative_error)
