from cudolfinx import cpp as _cucpp
from dolfinx import mesh

def ghost_layer_mesh(domain: mesh.Mesh):
    """Add a ghost layer of cells to the given mesh
    """
    _ghost_mesh = _cucpp.fem.ghost_layer_mesh(domain._cpp_object, domain._geometry._cpp_object.cmap)
    return mesh.Mesh(
            _ghost_mesh,
            domain._ufl_domain)

def ghost_layer_meshtags(meshtags: mesh.MeshTags, ghosted_mesh: mesh.Mesh):
    """Trasnfer meshtags to ghost layer mesh."""

    _cpp_meshtags = _cucpp.fem.ghost_layer_meshtags(meshtags._cpp_object, ghosted_mesh._cpp_object)
    return mesh.MeshTags(_cpp_meshtags)
