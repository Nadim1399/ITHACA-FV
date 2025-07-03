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
    std::cout << "qui1" << std::endl;

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
    Eigen::MatrixXd u_field = cnpy::load(temp_u, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/u_field_LSTM.npy");

    Eigen::MatrixXd temp_p;
    Eigen::MatrixXd p_field = cnpy::load(temp_p, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/p_field_LSTM.npy");

    Eigen::MatrixXd temp_n;
    Eigen::MatrixXd nut_field = cnpy::load(temp_n, "/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ROM_LSTM_Rec/nut_field_LSTM.npy");

    // Conversione npy to openfoam
    Foam2Eigen::Eigen2field(u_field_foam, u_field, true);
    Foam2Eigen::Eigen2field(p_field_foam, p_field, true);
    Foam2Eigen::Eigen2field(nut_field_foam, nut_field, true);

    Foam::PtrList<Foam::volVectorField> uFieldList(1);
    uFieldList.set(0, &u_field_foam);

    Foam::PtrList<Foam::volScalarField> pFieldList(1);
    pFieldList.set(0, &p_field_foam);

    Foam::PtrList<Foam::volScalarField> nutFieldList(1);
    nutFieldList.set(0, &nut_field_foam);

    Foam::word uOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ITHACAoutput/LSTM/u");
    Foam::word pOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ITHACAoutput/LSTM/p");
    Foam::word nutOutputFolder("/home/nrooho/ITHACA-FV/tutorials/CFD/19/19Prova/ITHACAoutput/LSTM/nut");

    Foam::word uFieldName("u");
    Foam::word pFieldName("p");
    Foam::word nutFieldName("nut");

    ITHACAstream::exportFields(uFieldList, uOutputFolder, uFieldName);
    ITHACAstream::exportFields(pFieldList, pOutputFolder, pFieldName);
    ITHACAstream::exportFields(nutFieldList, nutOutputFolder, nutFieldName);
}

int main(int argc, char *argv[])
{
    importAndExportFields(argc, argv);
    return 0;
}


