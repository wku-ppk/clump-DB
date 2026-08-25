import argparse

from clumpgen.shape import (
    SHAPE_FAMILIES,
    ShapeParams,
    make_particle_stl,
    source_shape_metrics,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/test.stl")
    ap.add_argument("--L", type=float, default=10.0)
    ap.add_argument("--e", type=float, default=0.8)
    ap.add_argument("--f", type=float, default=0.7)
    ap.add_argument("--shape-family", choices=SHAPE_FAMILIES, default="ellipsoid")
    ap.add_argument(
        "--taper",
        type=float,
        default=0.25,
        help="fractional reduction at one end; sign selects the end",
    )
    ap.add_argument(
        "--boxiness",
        type=float,
        default=3.0,
        help="superellipsoid exponent (2=ellipsoid, larger=boxier)",
    )
    ap.add_argument(
        "--asymmetry-I",
        type=float,
        default=0.0,
        help="signed one-sided bulging across the I direction",
    )
    ap.add_argument(
        "--asymmetry-S",
        type=float,
        default=0.0,
        help="signed one-sided bulging across the S direction",
    )
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--randomness", type=float, default=0.12)
    ap.add_argument("--bias", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    p = ShapeParams(
        L=args.L,
        e=args.e,
        f=args.f,
        family=args.shape_family,
        taper=args.taper,
        boxiness=args.boxiness,
        asymmetry_I=args.asymmetry_I,
        asymmetry_S=args.asymmetry_S,
        subdivisions=args.sub,
        randomness=args.randomness,
        bias=args.bias,
        seed=args.seed,
    )
    mesh = make_particle_stl(args.out, p)
    print("[OK] wrote:", args.out)
    print("[INFO] family =", args.shape_family)
    print(
        "[INFO] watertight =",
        mesh.is_watertight,
        "verts =",
        len(mesh.vertices),
        "faces =",
        len(mesh.faces),
    )
    print("[INFO] source metrics =", source_shape_metrics(mesh, p))


if __name__ == "__main__":
    main()
