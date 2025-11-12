/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Description
    Example of a heat transfer Reduction Problem
SourceFiles
    02thermalBlock.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "ITHACAutilities.H"
#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>

int main(int argc, char *argv[])
{

    // #include "setRootCase.H"
    // #include "createTime.H"
    // #include "createMesh.H"
    // #include "createFields.H"

    // PtrList<volVectorField> Ufield;
    // PtrList<volScalarField> Pfield;

    // ITHACAparameters* para = ITHACAparameters::getInstance(mesh,
    //                          runTime);
    // ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline");
    // ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline");

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
    ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
    ITHACAstream::read_fields(pfield, p, "./ITHACAoutput/Offline/");
    ITHACAstream::read_fields(nutfield, nut, "./ITHACAoutput/Offline/");
        
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
    
    Info << "Ufield: " << Ufield[3] << endl;
    Info << "pfield: " << pfield[3] << endl;
    Info << "nutfield: " << nutfield[3] << endl;

    Info << "Number of U snapshots: " << Ufield.size() << endl;
    Info << "Number of P snapshots: " << pfield.size() << endl;
    Info << "Number of nut snapshots: " << nutfield.size() << endl;

    Eigen::MatrixXd U_eig = Foam2Eigen::PtrList2Eigen(
        Ufield);
    // std::cout << "U_eig block (0-5,0-3):\n" << U_eig.block(0,0,5,3) << std::endl;

    Eigen::MatrixXd P_eig = Foam2Eigen::PtrList2Eigen(
        pfield);

    Eigen::MatrixXd Nut_eig = Foam2Eigen::PtrList2Eigen(
        nutfield);

    Info << "U.npy size: " << U_eig.rows() << " x " << U_eig.cols() << endl;
    Info << "P.npy size: " << P_eig.rows() << " x " << P_eig.cols() << endl;
    Info << "Nut.npy size: " << Nut_eig.rows() << " x " << Nut_eig.cols() << endl;

    cnpy::save(U_eig, "U.npy");
    cnpy::save(P_eig, "P.npy");
    cnpy::save(Nut_eig, "Nut.npy");
    return 0;
}                    
