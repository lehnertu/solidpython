import numpy as np
from copy import *
import vtk

# the minimum square distance for two points to be considered distinct
tolerance = np.square(0.01) 

# square distance of two points
def sqr_dist(p1,p2):
    return np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]) + np.square(p1[2]-p2[2])

# this returns the cosine of the angle between two adjacent edges
# if they are fully aligned it returns 1.0 going to -1.0 with increasing angle
def angle(pts):
    if len(pts) != 3: raise Exception("need 3 points to compute an angle.")
    v1 = pts[1]-pts[0]
    v1 = v1/np.linalg.norm(v1)
    v2 = pts[2]-pts[1]
    v2 = v2/np.linalg.norm(v2)
    return np.dot(v1,v2)

def VisualizePointCloud(points):
    """
    Display a set of points in 3D space
    """
    pts = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()
    for p in points:
        id = pts.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(id)
    meshData = vtk.vtkPolyData()
    meshData.SetPoints(pts)
    meshData.SetVerts(vertices)
    # map the triangle meshs into the scene
    meshMapper = vtk.vtkPolyDataMapper()
    meshMapper.SetInputData(meshData)
    # add the actors to the scene
    meshActor = vtk.vtkActor()
    meshActor.SetMapper(meshMapper)
    meshActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Yellow"))
    # create a render window
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("SlateGray"))
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(800,600)
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    renderWindowInteractor.Initialize()
    style = vtk.vtkInteractorStyleTrackballCamera()
    style.SetDefaultRenderer(renderer)
    renderWindowInteractor.SetInteractorStyle(style)
    # add the actors to the scene
    renderer.AddActor(meshActor)
    # render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()
    # now the interaction is running until we close the window
    # cleanup after closing the window
    del renderWindow
    del renderWindowInteractor

# We define a class for a polyhedron.
# Its starts empty and we can successively add faces keeping track of the number of points
# and the correct indices for faces.
# Finally we can output a solid geometry.
class SolidPolyhedron():
    def __init__(self):
        """
        Create an empty SolidPolyhedron object
        """
        self.points = []
        self.faces = []
        self.NP = 0
    def replace_point(self,old,new):
        """
        Replace all references to the old point by a reference to the new point.
        """
        for face in self.faces:
            for k,f in enumerate(face):
                if f==old:
                    # face is a reference, so we can modify it in place
                    face[k]=new
    def remove_point(self,index):
        """
        Remove one point with given index from the list.
        This is only possible if there exist no references to it in the faces.
        All face indexes are updated according to the shift in list positions.
        """
        # first check if there are no references
        for face in self.faces:
            for f in face:
                if f==index:
                    raise Exception("attempting to remove a point with existing references")
        # delete the point from the list
        del self.points[index]
        self.NP = len(self.points)
        # move the indexes of the faces
        for i in range(index,self.NP):
            self.replace_point(i+1,i)
    def add_triangle(self,new_points):
        """
        Add a triangle with given points to the faces of the solid.
        Points are only added if they are not yet present.
        """
        if len(new_points) != 3:
            raise Exception("triangles should be given with 3 points.")
        new_face = [0,0,0]
        # append the new points to the list
        for i,new in enumerate(new_points):
            is_new = True
            # check if this new point is already present
            for k,p in enumerate(self.points):
                if sqr_dist(p,new)<tolerance:
                    new_face[i] = k
                    is_new = False
            # do not append points that are already present
            if is_new:
                new_face[i] = self.NP
                self.points.append(new)
                self.NP += 1
        self.faces.append(new_face)        
    def add_polygon(self,new_points):
        """
        Add a face defined by a polygon.
        Degenerated edges are removed.
        The polygon is recursively split into triangles, always cutting off
        the triangle with the sharpest corner.
        """
        new_NP = len(new_points)
        # remove degenerate edges
        i=1
        # we have to use while loops as the end may change during the execution
        while i<new_NP:
            p1 = new_points[i-1]
            p2 = new_points[i]
            if sqr_dist(p1,p2)<tolerance:
                del new_points[i]
                new_NP -= 1
                print('removed one degenerate edge')
                # if the edge was degenerate we have to try te same index again
            else:
                i += 1
        # add the face
        if new_NP<3: raise Exception("too few points for a polygon.")
        if new_NP==3: self.add_triangle(new_points)
        else:
            # find the sharpest corner
            min_angle = 1.0
            # i is the index of the corner under consideration
            for i in range(new_NP):
                ind = [i-1,i,i+1]
                # the positive wrap-around has to be handled explicitely, the -1 index works as built-in
                if i+1==new_NP: ind = [i-1,i,0]
                points = [new_points[k] for k in ind]
                a = angle(points)
                if a<min_angle:
                    tri_ind = i
                    tri_points = points
                    min_angle = a
            self.add_triangle(tri_points)
            # the rest is the origonal polygon with the sharpest corner dropped
            rest_ind = range(new_NP)
            rest_ind.remove(tri_ind)
            rest = [new_points[i] for i in rest_ind]
            # recursively add the rest polygon
            self.add_polygon(rest)            
    def add(self,new_points,new_faces):
        """
        Add a collection of faces to the lists
        """
        old_NP = self.NP
        new_NP = len(new_points)
        # the new points are appended after the existing ones
        for p in new_points:
            self.points.append(p)
        # all indices have to be corrected for the number of already existing points
        for f in new_faces:
            new_face = [i+old_NP for i in f]
            self.faces.append(new_face)
        self.NP += new_NP
        # now check if any of the new points were already present
        # the code is the same as unify() except that the ranges of the test
        # are limited to the old (NP) and new points, respectively
        # we have to use while loops as the end may change during the execution
        i=0
        while i<old_NP:
            # k starts with the first new point
            k=old_NP
            while k<self.NP:
                if sqr_dist(self.points[i],self.points[k])<tolerance:
                    # replace the new point with the already present
                    self.replace_point(k,i)
                    self.remove_point(k)
                k+=1
            i+=1
    def unify(self):
        """
        Check for duplicated points with less than tolerance distance.
        """
        # we have to use while loops as the end may change during the execution
        i=0
        while i<self.NP:
            k=i+1
            while k<self.NP:
                if sqr_dist(self.points[i],self.points[k])<tolerance:
                    # replace the latter point with the former
                    self.replace_point(k,i)
                    self.remove_point(k)
                k+=1
            i+=1
    def summary(self):
        print("NP = %d" % self.NP)
        print("faces %d" % len(self.faces))
    def check(self,debug=False):
        """
        We want to check the correctness and completeness of the solid.
        It is a closed volume with all normals pointing to the same side if all edges
        exit exactly twice with opposite directions.
        The debug flag controls the generation of output about every flaw found.
        """
        # make a list of all edges starting from the points
        count = 0
        edges = [[] for i in range(self.NP)]
        for f in self.faces:
            NF = len(f)
            for i in range(NF-1):
                edges[f[i]].append(f[i+1])
                count += 1
            edges[f[NF-1]].append(f[0])
            count += 1
        print('found %d edges' % count)
        # check for duplicated edges
        count = 0
        for p1,e in enumerate(edges):
            set_e = set()
            for p2 in e:
                if p2 in set_e:
                    if debug: print('found duplicated edge from %d to %d.' % (p1,p2))
                    count += 1
                else:
                    set_e.add(p2)
        print('found %d duplicated edges' % count)
        # check for every edge if the opposite direction exists
        count = 0
        for p1 in range(self.NP):
            for p2 in edges[p1]:
                if not p1 in edges[p2]:
                    count = count+1
                    if debug: print('found free edge from %d to %d.' % (p1,p2))
        print('found %d free edges' % count)
    def getPolyData(self):
        """
        Return a vtkPolyData object
        """
        pts = vtk.vtkPoints()
        for p in self.points: pts.InsertNextPoint(p)
        cells = vtk.vtkCellArray()
        for f in self.faces:
            cells.InsertNextCell(len(f), f)
        meshData = vtk.vtkPolyData()
        meshData.SetPoints(pts)
        meshData.SetPolys(cells)
        return meshData
    def writeSTL(self,filename):
        pts = vtk.vtkPoints()
        for p in self.points: pts.InsertNextPoint(p)
        cells = vtk.vtkCellArray()
        for f in self.faces:
            cells.InsertNextCell(len(f), f)
        meshData = vtk.vtkPolyData()
        meshData.SetPoints(pts)
        meshData.SetPolys(cells)
        stlWriter = vtk.vtkSTLWriter()
        stlWriter.SetFileTypeToASCII()
        # stlWriter.SetFileTypeToBinary()
        stlWriter.SetInputData(meshData)
        stlWriter.SetFileName(filename)
        stlWriter.Write()
    def visualize(self, showEdges=True, Opacity=0.9):
        meshData = self.getPolyData()
        # map the triangle meshs into the scene
        meshMapper = vtk.vtkPolyDataMapper()
        meshMapper.SetInputData(meshData)
        # add the actors to the scene
        meshActor = vtk.vtkActor()
        meshActor.SetMapper(meshMapper)
        if showEdges:
            meshActor.GetProperty().EdgeVisibilityOn()
        else:
            meshActor.GetProperty().EdgeVisibilityOff()
        meshActor.GetProperty().SetColor(vtk.vtkNamedColors().GetColor3d("Yellow"))
        meshActor.GetProperty().SetOpacity(Opacity)
        # create a render window
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("SlateGray"))
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetSize(800,600)
        renderWindow.AddRenderer(renderer)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.Initialize()
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.SetDefaultRenderer(renderer)
        renderWindowInteractor.SetInteractorStyle(style)
        # add the actors to the scene
        renderer.AddActor(meshActor)
        # render and interact
        renderWindow.Render()
        renderWindowInteractor.Start()
        # now the interaction is running until we close the window
        # cleanup after closing the window
        del renderWindow
        del renderWindowInteractor      