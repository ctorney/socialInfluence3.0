
#ifndef _plotGrid_h
#define _plotGrid_h
#include <vtkVersion.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleImage.h>
#include <vtkRenderer.h>
#include <vtkImageMapper.h>
#include <vtkActor2D.h>
#include <vtkMath.h>
class plotGrid {
    public:
        plotGrid();
        void draw(int Nx, int *states);
    protected:
        vtkSmartPointer<vtkImageData> image;
        vtkSmartPointer<vtkImageMapper> imageMapper;
        vtkSmartPointer<vtkActor2D> imageActor;
        vtkSmartPointer<vtkRenderer> renderer;
        vtkSmartPointer<vtkRenderWindow> renderWindow;
        vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;
        vtkSmartPointer<vtkInteractorStyleImage> style;
};
#endif
