import mteb
from collections import defaultdict

bench = mteb.get_benchmark("MTEB(eng, classic)")
evaluator = mteb.MTEB(tasks=bench)


def tname(t):
    return getattr(
        t,
        "task_name",
        getattr(getattr(t, "metadata", None), "name", t.__class__.__name__),
    )


def ttype(t):
    return getattr(
        t, "task_type", getattr(getattr(t, "metadata", None), "type", "Unknown")
    )


by_type = defaultdict(list)
for t in evaluator.tasks:
    by_type[ttype(t)].append(tname(t))


if __name__ == "__main__":
    # total = 0
    # for k in sorted(by_type):
    #     items = sorted(set(by_type[k]))
    #     total += len(items)
    #     print(f"\n=== {k} ({len(items)}) ===")
    #     for s in items:
    #         print(s)
    # print(f"\nTotal datasets in benchmark: {total}")

    print(by_type.get("Classification"))  # Example usage
