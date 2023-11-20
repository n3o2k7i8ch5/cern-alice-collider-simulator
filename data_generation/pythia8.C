/// \file
/// \ingroup tutorial_pythia
/// pythia8 basic example
///
/// to run, do:
///
/// ~~~{.cpp}
///  root > .x pythia8.C
/// ~~~
///
/// \macro_code
///
/// \author Andreas Morsch

#include <iostream>


#include "TSystem.h"
#include "TH1F.h"
#include "TClonesArray.h"
#include "TPythia8.h"
#include "TParticle.h"
#include "TDatabasePDG.h"
#include "TCanvas.h"

#include "time.h"

#define FILENAME "pythia.root"
#define TREENAME "tree"
#define BRANCHNAME "particles"

void pythia8(Int_t nev = 10000000, Int_t ndeb = 100)
{
    clock_t start = clock();
    
// Load libraries
   gSystem->Load("libEG");
   gSystem->Load("libEGPythia8");
// Histograms
   TH1F* etaH = new TH1F("etaH", "Pseudorapidity", 120, -12., 12.);
   TH1F* ptH  = new TH1F("ptH",  "pt",              50,   0., 10.);


// Array of particles
   TClonesArray* particles = new TClonesArray("TParticle", 1000);
// Create pythia8 object
   TPythia8* pythia8 = new TPythia8();

// Configure
   pythia8->ReadString("HardQCD:all = on");
   pythia8->ReadString("Random:setSeed = on");
   // use a reproducible seed: always the same results for the tutorial.
   pythia8->ReadString("Random:seed = 42");


// Initialize

   pythia8->Initialize(2212 /* p */, 2212 /* p */, 14000. /* TeV */);

   //-----------------------------------
   //-----------------------------------
   //-----------------------------------
      
    TFile* file = TFile::Open(FILENAME, "RECREATE");
    if (!file || !file->IsOpen()) {
        Error("makeEventSample", "Couldn;t open file %s", FILENAME);
        return 1;
    }
    TTree* tree = new TTree(TREENAME, "Pythia 8 tree");
   
    particles = (TClonesArray*)pythia8->GetListOfParticles();
    tree->Branch(BRANCHNAME, &particles);
    
// Event loop
    
    int total_np = 0;
    
   for (Int_t iev = 0; iev < nev; iev++) {
      pythia8->GenerateEvent();
      if (iev < ndeb) pythia8->EventListing();
      pythia8->ImportParticles(particles,"All");

      Int_t np = particles->GetEntriesFast();
      std::cout << iev << " " << np << std::endl;
      
      total_np += np;
      if (total_np >= nev){
         iev = 1000000000; 
        }
      
// Particle loop
      for (Int_t ip = 0; ip < np; ip++) {
          //tree->Fill();
          
         TParticle* part = (TParticle*) particles->At(ip);
         Int_t ist = part->GetStatusCode();
         // Positive codes are final particles.
         if (ist <= 0) continue;
         Int_t pdg = part->GetPdgCode();
         Float_t charge = TDatabasePDG::Instance()->GetParticle(pdg)->Charge();
         if (charge == 0.) continue;
         Float_t eta = part->Eta();
         Float_t pt  = part->Pt();

         etaH->Fill(eta);
         if (pt > 0.) ptH->Fill(pt, 1./(2. * pt));
         
      }
      
      tree->Fill();
   }

   //TVirtualMCStack* mcStack = new TVirtualMCStack()
      
   //pythia8->PrintStatistics();
   
   clock_t end = clock();
   
   printf("%f\n", (float)(end - start) / CLOCKS_PER_SEC);
   
   file->Write();
   file->Close();
   
 }
