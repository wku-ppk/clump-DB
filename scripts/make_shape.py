import argparse
from clumpgen.shape import ShapeParams, make_irregular_ellipsoid_stl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/test.stl")
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--e", type=float, default=0.8)
    ap.add_argument("--f", type=float, default=0.7)
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--randomness", type=float, default=0.12)
    ap.add_argument("--bias", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    p = ShapeParams(
        L=args.L, e=args.e, f=args.f,
        subdivisions=args.sub,
        randomness=args.randomness,
        bias=args.bias,
        seed=args.seed,
    )
    mesh = make_irregular_ellipsoid_stl(args.out, p)
    print("[OK] wrote:", args.out)
    print("[INFO] watertight =", mesh.is_watertight, "verts =", len(mesh.vertices), "faces =", len(mesh.faces))

if __name__ == "__main__":
    main()