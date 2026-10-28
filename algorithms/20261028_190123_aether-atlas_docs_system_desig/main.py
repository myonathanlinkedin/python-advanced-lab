import unittest
from typing import List, Dict, Union
from fractions import Fraction

# Import the CRDT implementation
from core import TextCRDT


def exchange_ops(replica_a: TextCRDT, replica_b: TextCRDT,
                 ops_a: List[Dict[str, Union[str, int, Fraction]]],
                 ops_b: List[Dict[str, Union[str, int, Fraction]]]) -> None:
    """
    Simulate a bidirectional network exchange of operation logs.
    Each operation is applied to the opposite replica.
    """
    for op in ops_a:
        replica_b.apply_remote(op)
    for op in ops_b:
        replica_a.apply_remote(op)


class TestTextCRDT(unittest.TestCase):
    def test_basic_insert_delete(self):
        crdt = TextCRDT(site_id="A")
        # Insert "hello"
        ops = []
        for i, ch in enumerate("hello"):
            ops.append(crdt.local_insert(i, ch))
        self.assertEqual(str(crdt), "hello")
        # Delete 'e' (index 1)
        del_op = crdt.local_delete(1)
        ops.append(del_op)
        self.assertEqual(str(crdt), "hllo")
        # Apply same ops to a fresh replica
        replica = TextCRDT(site_id="B")
        for op in ops:
            replica.apply_remote(op)
        self.assertEqual(str(replica), "hllo")

    def test_concurrent_inserts(self):
        # Two replicas start empty
        a = TextCRDT(site_id="A")
        b = TextCRDT(site_id="B")
        # Replica A inserts "A" at position 0
        op_a1 = a.local_insert(0, "A")
        # Replica B inserts "B" at position 0 concurrently
        op_b1 = b.local_insert(0, "B")
        # Exchange operations
        exchange_ops(a, b, [op_a1], [op_b1])
        # Both replicas should now contain both characters in a deterministic order
        self.assertEqual(str(a), str(b))
        self.assertIn("AB", str(a) + str(b))  # order may be AB or BA depending on keys
        # Verify that ordering respects the generated keys
        positions_a = a.to_list()
        self.assertLess(positions_a[0][0], positions_a[1][0])  # keys are ordered

    def test_complex_scenario(self):
        # Simulate three replicas editing the same document
        replicas = [TextCRDT(site_id=chr(ord('A') + i)) for i in range(3)]
        ops: List[List[Dict[str, Union[str, int, Fraction]]]] = [[] for _ in replicas]

        # Replica 0 inserts "cat"
        for i, ch in enumerate("cat"):
            ops[0].append(replicas[0].local_insert(i, ch))

        # Replica 1 inserts "dog" at the beginning concurrently
        for i, ch in enumerate("dog"):
            ops[1].append(replicas[1].local_insert(i, ch))

        # Replica 2 inserts "fish" after the first character of its own view (still empty)
        ops[2].append(replicas[2].local_insert(0, "f"))
        ops[2].append(replicas[2].local_insert(1, "i"))
        ops[2].append(replicas[2].local_insert(2, "s"))
        ops[2].append(replicas[2].local_insert(3, "h"))

        # Pairwise exchange of all operation logs
        for i in range(3):
            for j in range(i + 1, 3):
                exchange_ops(replicas[i], replicas[j], ops[i], ops[j])

        # After full synchronization, all replicas must converge to the same string
        final_strings = {str(r) for r in replicas}
        self.assertEqual(len(final_strings), 1)
        final_text = final_strings.pop()
        # The exact order is deterministic but not predetermined; ensure all characters are present
        for ch in "catdogfish":
            self.assertIn(ch, final_text)

    def test_delete_then_insert_same_position(self):
        crdt = TextCRDT(site_id="A")
        op1 = crdt.local_insert(0, "X")
        op2 = crdt.local_delete(0)
        # Insert a new character at the same logical index after deletion
        op3 = crdt.local_insert(0, "Y")
        # Apply ops to a fresh replica
        replica = TextCRDT(site_id="B")
        for op in (op1, op2, op3):
            replica.apply_remote(op)
        self.assertEqual(str(replica), "Y")

    def test_idempotent_remote_application(self):
        crdt = TextCRDT(site_id="A")
        ops = [crdt.local_insert(0, "Z")]
        # Apply the same operation twice; state must remain unchanged
        crdt.apply_remote(ops[0])
        crdt.apply_remote(ops[0])
        self.assertEqual(str(crdt), "Z")


if __name__ == "__main__":
    unittest.main()
