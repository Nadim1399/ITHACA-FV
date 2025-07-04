#include "ITHACAstream.H"
#include "Foam2Eigen.H"
#include "fvMesh.H"
#include "fvCFD.H"
#include "dynamicFvMesh.H"

void importAndExportFields(int argc, char *argv[])
{
    
    Foam::argList args(argc, argv);
    Foam::fileName caseDir(".");
    
    Foam::Time runTime(
        Foam::Time::controlDictName,args
    );
    
    Foam::fvMesh mesh(
        Foam::IOobject("region0",
            runTime.constant(),
            runTime,
            Foam::IOobject::MUST_READ)
        );
        
        std::cout << "qui1" << std::endl; //
        
        Foam::IOobject uIO
        (
            "U",                              
            runTime.timeName(),               
            mesh,                             
            Foam::IOobject::MUST_READ,         
            Foam::IOobject::AUTO_WRITE         
        );
        
        Foam::volVectorField u_field_foam(uIO, mesh);    
        
        std::cout << "qui2" << std::endl; //
        
        Foam::IOobject pIO
        (
            "p",                              
            runTime.timeName(),               
            mesh,                             
            Foam::IOobject::MUST_READ,         
            Foam::IOobject::AUTO_WRITE         
        );
        
        Foam::volScalarField p_field_foam(pIO, mesh);
        
        std::cout << "qui3" << std::endl; //
        
        Foam::IOobject nutIO
        (
            "nut",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        );
        
        Foam::volScalarField nut_field_foam(nutIO, mesh);
        
        std::cout << "qui3" << std::endl; //
        
        
        Eigen::MatrixXd temp_u;
        // Eigen::MatrixXd u_field = cnpy::load(temp_u, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/u_field_LSTM.npy");
        Eigen::MatrixXd u_field = cnpy::load(temp_u, "corrected.npy");
        
        Eigen::MatrixXd temp_p;
        Eigen::MatrixXd p_field = cnpy::load(temp_p, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/p_field_LSTM.npy");
        
        Eigen::MatrixXd temp_n;
        Eigen::MatrixXd nut_field = cnpy::load(temp_n, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/nut_field_LSTM.npy");
        
        std::cout << "u_field shape: " << u_field.rows() << " x " << u_field.cols() << std::endl;
        std::cout << "p_field shape: " << p_field.rows() << " x " << p_field.cols() << std::endl;
        std::cout << "nut_field shape: " << nut_field.rows() << " x " << nut_field.cols() << std::endl;

        PtrList<volVectorField> snapshots_u;
        
        for(int i = 0; i<u_field.cols();i++)
        {
            Eigen::VectorXd u_tmp = u_field.col(i);
            u_field_foam = Foam2Eigen::Eigen2field(u_field_foam, u_tmp, true);
            snapshots_u.append(u_field_foam.clone());
        }
        
        PtrList<volScalarField> snapshots_p;

        for(int i = 0; i<p_field.cols(); i++)
        {
            Eigen::VectorXd p_tmp = p_field.col(i);
            p_field_foam = Foam2Eigen::Eigen2field(p_field_foam, p_tmp, true);
            snapshots_p.append(p_field_foam.clone());
        }

        PtrList<volScalarField> snapshots_nut;

        for(int i = 0; i<nut_field.cols(); i++)
        {
            Eigen::VectorXd nut_tmp = nut_field.col(i);
            nut_field_foam = Foam2Eigen::Eigen2field(nut_field_foam, nut_tmp, true);
            snapshots_nut.append(nut_field_foam.clone());
        }
        
        Foam::word uOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ITHACAoutput/LSTM/u");
        Foam::word pOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ITHACAoutput/LSTM/p");
        Foam::word nutOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ITHACAoutput/LSTM/nut");
        
        Foam::word uFieldName("u");
        Foam::word pFieldName("p");
        Foam::word nutFieldName("nut");
        
        ITHACAstream::exportFields(snapshots_u, uOutputFolder, uFieldName);
        ITHACAstream::exportFields(snapshots_p, pOutputFolder, pFieldName);
        ITHACAstream::exportFields(snapshots_nut, nutOutputFolder, nutFieldName);
    }
    
    int main(int argc, char *argv[])
    {
        importAndExportFields(argc, argv);
        return 0;
    }
    
    
    