========================
Traced value provenance
========================

This page is for people changing LEAPP's tracer. It describes how a traced
value knows which FX graph value represents it. Nothing here is part of the
public API, and none of it is needed to annotate a pipeline.

One proxy per traced value
==========================

Every traced carrier --- ``TracedTensor``, ``TracedNpArray``, and
``TracedWpArray`` --- pairs a real eager value with the ``torch.fx.Proxy`` that
represents that value in the node's graph. Operations read the proxy, record a
new FX node, and hand the result proxy to the carrier they return:

.. code-block:: python

   x = ctx.create_input(torch.randn(4), name="x")   # x.proxy is a placeholder
   y = x + 1                                        # y.proxy is the add node

Read the proxy through the ``proxy`` property. It is read-only from outside the
carrier, and it is ``None`` when the value carries no graph provenance, which is
the normal state outside an active trace.

The ``ProxyView`` indirection
=============================

Carriers do not store the proxy directly. They store a ``ProxyView``, which
holds one mutable reference to the proxy that currently represents the value.
``carrier.proxy`` reads ``ProxyView.proxy``.

The indirection exists because of shared memory. Torch, NumPy, and Warp can all
expose several logical arrays over one allocation, and eager code mutates that
memory in place:

.. code-block:: python

   base = torch.tensor([1.0, 2.0, 3.0, 4.0])
   left = base[0:2]
   left += 1                 # base is now [2.0, 3.0, 3.0, 4.0]

An FX proxy identifies one immutable graph value, so a mutation cannot be
recorded as a mutation; it is recorded functionally and the carrier is pointed
at the new result. When several carriers share memory, they must end up pointing
at the same updated value, and a reference that other objects can observe is
what makes that possible. ``ProxyView`` is where that behavior is implemented,
so proxy readers do not have to know about aliasing.

Today every ``ProxyView`` is a root: it owns a proxy directly, reading it is a
plain attribute access, and no two carriers share one.

Creating versus replacing
=========================

There are two ways to give a carrier a proxy, and they are not
interchangeable.

**Create a new view** with ``_init_tracing_state()`` when the value itself is
new: a fresh carrier, an out-of-place operation result, or a value re-wrapped at
a node boundary. This also resets the carrier's name and clears its output port,
since a new value has not been published by any node.

**Replace the root** by assigning ``self._proxy_view.proxy`` when an existing
value was mutated in place: augmented arithmetic, an ``add_``-style method,
``copy_()``, the functional lowering of indexed assignment, or the
mutated-receiver path in ``__torch_function__``. The value keeps its identity and
only its representation changes, so nothing else about the carrier is touched.

Picking the wrong one is not currently observable, because no view is shared.
It becomes observable as soon as aliases share a view, so treat the distinction
as load-bearing.

Current limitations
===================

Two consequences of the root-only model are worth knowing before you rely on
mutation through an alias:

* Two carriers over the same memory hold independent views. Mutating one is not
  visible through the other, so the second keeps recording from its earlier
  proxy.
* A ``TracedNpArray`` view inherits its base's proxy, so mutating the view
  records the base's value rather than the slice.

Both are addressed by turning a view into a child of another view, which stores
how to project a parent value into the child and how to propagate an updated
child back to the parent. Reading a child then walks to the root and projects
forward. That work is deliberately separate from the indirection itself.
