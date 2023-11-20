#include <iostream>

#include <TApplication.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TF1.h>
#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TParticle.h>
#include <TH1D.h>
#include <TList.h>
#include <TKey.h>
#include <TDirectoryFile.h>
#include <TStopwatch.h>
#include <TStyle.h>
#include <TClonesArray.h>

#include <stdexcept>

/*
 Int_t          fPdgCode;              // PDG code of the particle
 Int_t          fStatusCode;           // generation status code    0 lub 1 (?)
 Int_t          fMother[2];            // Indices of the mother particles
 Int_t          fDaughter[2];          // Indices of the daughter particles
 Float_t        fWeight;               // particle weight

 Double_t       fCalcMass;             // Calculated mass

 Double_t       fPx;                   // x component of momentum
 Double_t       fPy;                   // y component of momentum
 Double_t       fPz;                   // z component of momentum
 Double_t       fE;                    // Energy

 Double_t       fVx;                   // x of production vertex
 Double_t       fVy;                   // y of production vertex
 Double_t       fVz;                   // z of production vertex
 Double_t       fVt;                   // t of production vertex

 Double_t       fPolarTheta;           // Polar angle of polarisation
 Double_t       fPolarPhi;             // azymutal angle of polarisation

 TParticlePDG*  fParticlePDG;          //! reference to the particle record in PDG database
 */

using namespace std;

int main(int argc, char **argv)
{
    TFile f("pythia.root");
    TTree* t = (TTree*)f.Get("tree");
    TClonesArray *part = nullptr;
    Long_t i=0, j=0;
    t->SetBranchAddress("particles", &part);
    
    Long_t entry_count = t->GetEntries();
        
    for (i=0; i<entry_count; i++) {
        t->GetEntry(i);
        Long_t part_entry_count = part->GetEntries();
        for (j=0; j<part_entry_count; j++) {

            TParticle* p = (TParticle*)part->At(j);
            // here p contains particle nr j from run i.
            //printf("Run: %d, Particle nr %d is %s\n",i,j,p->GetName());

  
            Int_t pdg_code = p->GetPdgCode();
            //https://root-forum.cern.ch/t/unknown-pid-9942003-pdg-info-will-be-ignored-e-g-charge-root/29499
            if(pdg_code == 9942003 ||
                pdg_code == 9941003 ||
                pdg_code == 9010221 ||
                pdg_code == 9942033 ||
                pdg_code == 30443 ||
                pdg_code == 9942103 ||
                pdg_code == 4124 ||
                pdg_code == 9940003
            ) continue;
            
            cout << pdg_code << " ";                //0     PDG code of the particle
            cout << p->GetStatusCode() << " ";      //1     generation status code
            cout << p->GetFirstMother() << " ";     //2     Indices of the mother particles
            cout << p->GetSecondMother() << " ";    //3
            cout << p->GetFirstDaughter() << " ";   //4     Indices of the daughter particles
            cout << p->GetLastDaughter() << " ";  //5

            cout << p->GetWeight() << " ";          //6     particle weight
            
            //cout << p->GetCalcMass() << " ";        //7     Calculated mass    to czasem z niewyjaśnionych powodów generuje segmentation violation
            cout << p->Px() << " ";                 //8     x component of momentum
            cout << p->Py() << " ";                 //9     y component of momentum
            cout << p->Pz() << " ";                 //10    z component of momentum
            cout << p->Energy() << " ";             //11    Energy
            cout << p->Vx() << " ";                 //12    x of production vertex
            cout << p->Vy() << " ";                 //13    y of production vertex
            cout << p->Vz() << " ";                 //14    z of production vertex
            cout << p->GetPolarTheta() << " ";      //15    Polar angle of polarisation
            cout << p->GetPolarPhi() << " ";        //16    azymutal angle of polarisation
            cout << endl;
            //TParticlePDG* pdg = p->GetPDG();
            //pdg->
            //printf("%f ", p->GetPDG());
            
        }

        cout << endl;
    }

    return 0;
}
