* Auto-generated simple SPICE-like netlist

C1 n0 0 1u
* D1 type=dc supply nets=n1
* D2 type=dc supply nets=n2
* G1 type=ground nets=n3
R1 n4 0 1k
I1 n5 0 1m
I2 n2 0 1m
R2 n0 0 1k
* I1 type=independent dc current nets=n3
R3 n6 0 1k

.end
