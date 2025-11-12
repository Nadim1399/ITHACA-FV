#include "ITHACAstream.H"
#include "Foam2Eigen.H"
#include "fvMesh.H"
#include "fvCFD.H"
#include "dynamicFvMesh.H"


void importAndExportFields(int argc, char *argv[])
{    

    PtrList<volVectorField> Ufield;
    word U("U");

    PtrList<volScalarField> pfield;
    word p("p");

    PtrList<volScalarField> nutfield;
    word nut("nut");

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

    ITHACAparameters::getInstance(mesh, runTime);
    ITHACAstream::read_fields(Ufield, U, "./Offline/");
    ITHACAstream::read_fields(pfield, p, "./Offline/");
    ITHACAstream::read_fields(nutfield, nut, "./Offline/");
        
        Foam::IOobject uIO
        (
            "U",                              
            runTime.timeName(),               
            mesh,                             
            Foam::IOobject::MUST_READ,         
            Foam::IOobject::AUTO_WRITE         
        );
        
        Foam::volVectorField u_field_foam(uIO, mesh);    

        Foam::IOobject pIO
        (
            "p",                              
            runTime.timeName(),               
            mesh,                             
            Foam::IOobject::MUST_READ,         
            Foam::IOobject::AUTO_WRITE         
        );
        
        Foam::volScalarField p_field_foam(pIO, mesh);
        
        Foam::IOobject nutIO
        (
            "nut",
            runTime.timeName(),
            mesh,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::AUTO_WRITE
        );
        
        Foam::volScalarField nut_field_foam(nutIO, mesh);

        // // Converte volVectorField in Eigen::MatrixXd
        // Eigen::MatrixXd U_FOM = Foam2Eigen::field2Eigen(u_field_foam); // shape: nCells x 3 (o 1 se monodimensionale)
        // Eigen::MatrixXd P_FOM = Foam2Eigen::field2Eigen(p_field_foam); // shape: nCells x 1
        // Eigen::MatrixXd NUT_FOM = Foam2Eigen::field2Eigen(nut_field_foam); // shape: nCells x 1

        // std::cout << "DEBUG: U_FOM shape = " << U_FOM.rows() << " x " << U_FOM.cols() << std::endl;
        // std::cout << "DEBUG: P_FOM shape = " << P_FOM.rows() << " x " << P_FOM.cols() << std::endl;
        // std::cout << "DEBUG: NUT_FOM shape = " << NUT_FOM.rows() << " x " << NUT_FOM.cols() << std::endl;


        // cnpy::save(U_FOM, "./FOM_U.npy");
        // cnpy::save(P_FOM, "./FOM_p.npy");
        // cnpy::save(NUT_FOM, "./FOM_nut.npy");

        ///////// LSTM /////////
        // Eigen::MatrixXd temp_u;
        // Eigen::MatrixXd u_field = cnpy::load(temp_u, "corrected.npy");
        
        // Eigen::MatrixXd temp_p;
        // Eigen::MatrixXd p_field = cnpy::load(temp_p, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/p_field_LSTM.npy");
        
        // Eigen::MatrixXd temp_n;
        // Eigen::MatrixXd nut_field = cnpy::load(temp_n, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/nut_field_LSTM.npy");
        
        ///////// MLP /////////
        // Eigen::MatrixXd temp_u;
        // Eigen::MatrixXd u_field = cnpy::load(temp_u, "corrected.npy");
        
        // Eigen::MatrixXd temp_p;
        // Eigen::MatrixXd p_field = cnpy::load(temp_p, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19MLP/MLP_Rec/p_field_MLP.npy");
        
        // Eigen::MatrixXd temp_n;
        // Eigen::MatrixXd nut_field = cnpy::load(temp_n, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19MLP/MLP_Rec/nut_field_MLP.npy");
        
        ///////// TRANSFORMER /////////
        Eigen::MatrixXd temp_u;
        Eigen::MatrixXd u_field = cnpy::load(temp_u, "corrected.npy");
        
        Eigen::MatrixXd temp_p;
        Eigen::MatrixXd p_field = cnpy::load(temp_p, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Transformer/Tr_Rec/p_field_Tr.npy");
        
        Eigen::MatrixXd temp_n;
        Eigen::MatrixXd nut_field = cnpy::load(temp_n, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Transformer/Tr_Rec/nut_field_Tr.npy");
        

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
        
        Foam::word uOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Transformer/ITHACAoutput/Tr/u");
        Foam::word pOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Transformer/ITHACAoutput/Tr/p");
        Foam::word nutOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Transformer/ITHACAoutput/Tr/nut");
        
        Foam::word uFieldName("u");
        Foam::word pFieldName("p");
        Foam::word nutFieldName("nut");
        
        ITHACAstream::exportFields(snapshots_u, uOutputFolder, uFieldName);
        ITHACAstream::exportFields(snapshots_p, pOutputFolder, pFieldName);
        ITHACAstream::exportFields(snapshots_nut, nutOutputFolder, nutFieldName);

        // --- DEBUG dimensioni snapshot ---
        std::cout << "DEBUG: snapshots_u_FOM.size() = " << Ufield.size() << std::endl;
        std::cout << "DEBUG: snapshots_u.size() = " << snapshots_u.size() << std::endl;

        std::cout << "DEBUG: snapshots_p_FOM.size() = " << pfield.size() << std::endl;
        std::cout << "DEBUG: snapshots_p.size() = " << snapshots_p.size() << std::endl;

        std::cout << "DEBUG: snapshots_nut_FOM.size() = " << nutfield.size() << std::endl;
        std::cout << "DEBUG: snapshots_nut.size() = " << snapshots_nut.size() << std::endl;


        Eigen::MatrixXd Errors_U;
        Errors_U = ITHACAutilities::errorL2Rel(Ufield, snapshots_u);
        cnpy::save(Errors_U, "./Tr/error_U.npy");

        Eigen::MatrixXd Errors_p;
        Errors_p = ITHACAutilities::errorL2Rel(pfield, snapshots_p);
        cnpy::save(Errors_p, "./Tr/error_p.npy");

        Eigen::MatrixXd Errors_nut;
        Errors_nut = ITHACAutilities::errorL2Rel(nutfield, snapshots_nut);
        cnpy::save(Errors_nut, "./Tr/error_nut.npy");
    }
    
    int main(int argc, char *argv[])
    {
        importAndExportFields(argc, argv);
        return 0;
    }
