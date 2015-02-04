#include "plotGrid.h"

#define WIN_SIZE 600
#define BORDER 20
plotGrid::plotGrid()
{
  image = vtkSmartPointer<vtkImageData>::New();
  imageActor = vtkSmartPointer<vtkActor2D>::New();
  renderer = vtkSmartPointer<vtkRenderer>::New();
  renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
  renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  style = vtkSmartPointer<vtkInteractorStyleImage>::New();
  imageMapper = vtkSmartPointer<vtkImageMapper>::New();

  imageMapper->SetInputConnection(image->GetProducerPort());
  imageMapper->SetColorWindow(1);
  imageMapper->SetColorLevel(0.5);
  
  imageActor->SetMapper(imageMapper);
        
  renderWindow->AddRenderer(renderer);
  int bSize = 2*BORDER + WIN_SIZE;
  renderWindow->SetSize(bSize,bSize); //(width, height)
  //renderWindowInteractor->SetRenderWindow(renderWindow);
  renderer->AddActor2D(imageActor);
  
  return;
}

void plotGrid::draw(int Nx, int* states)
{
  unsigned int block = WIN_SIZE/Nx;
  
  image->SetDimensions(Nx*block, Nx*block, 1);
  image->SetNumberOfScalarComponents(1);
  image->SetScalarTypeToDouble();
  image->AllocateScalars();
  for(unsigned int x = 0; x < Nx; x++)
      for(unsigned int y = 0; y < Nx; y++)
          for(unsigned int bx = 0; bx < block; bx++)
              for(unsigned int by = 0; by < block; by++)
              {
                  double* pixel = static_cast<double*>(image->GetScalarPointer(x*block+bx,y*block+by,0));
                  pixel[0] = (1.0)*states[Nx*y + x];
              }
   
    
  image->Modified();
  renderWindow->Render();
  renderer->ResetCamera();
  imageActor->SetPosition(BORDER, BORDER);
  //renderWindow->SetSize( 600, 600 );
  //renderer->ResetCamera(0,1,0,1,0,1);
  //renderer->GetActiveCamera()->Zoom(1.5);

  //renderWindowInteractor->Start();
}
