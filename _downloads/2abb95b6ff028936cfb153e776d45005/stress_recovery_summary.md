# Local stress recovery

Loads: M1 = 50 N m, M2 = 100 N m. Stress components are in MPa.

| Component | Minimum | Maximum |
|---|---:|---:|
| S11 | -35.625 | +35.625 |
| S22 | -2.129e-6 | +1.899e-6 |
| S33 | -7.646e-7 | +7.099e-7 |
| S23 | -3.766e-7 | +4.061e-7 |
| S13 | -9.423 | +9.423 |
| S12 | -12.027 | +12.027 |

Analytical bending stress: M2 c / I2 = 37.500 MPa, where c = 0.010 m and I2 = 2.6667e-8 m^4. The reported OpenSG peak is an element mean at y3 = +-9.5 mm, giving 35.625 MPa. Its expected analytical value at that sampling location is 35.625 MPa; it agrees to displayed precision. The physical extreme-fibre stress is 37.500 MPa.

Recovered resultants: M1 = 50.039 N m (0.0776% high), M2 = 100.071 N m (0.0714% high).
