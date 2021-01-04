//Définition des valeurs physiques
L = 65e-3; //Longeur selon x
H = 120e-3; //Longuer selon y
l0 = 10e-3; //Size initial crack
h = 1e-3; //1.5e-3; //2e-3 //Taille du maillage

//Plate
Point(1) = {0,H/2,0,h};
Point(2) = {0,H/2-55e-3,0,h};
Point(3) = {0,-H/2,0,h};
Point(4) = {L,-H/2,0,h};
Point(5) = {L,H/2,0,h};
Point(6) = {l0,H/2-55e-3,0,h};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,1};
Line(6) = {2,6};

Line Loop(1) = {1:5};

//Hole 1
radius = 5e-3;
Point(7) = {20e-3, H/2-20e-3, 0, h}; //Centre of the hole
Point(8) = {20e-3+radius, H/2-20e-3, 0, h};
Point(9) = {20e-3-radius, H/2-20e-3, 0, h};
Point(10) = {20e-3, H/2-20e-3+radius, 0, h};
Point(11) = {20e-3, H/2-20e-3-radius, 0, h};

Circle(7) = {10, 7, 9};
Circle(8) = {9, 7, 11};
Circle(9) = {11, 7, 8};
Circle(10) = {8, 7, 10};
Line Loop(2) = {7:10};

//Hole 2
Point(12) = {20e-3, -H/2+20e-3, 0, h}; //Centre of the hole
Point(13) = {20e-3+radius, -H/2+20e-3, 0, h};
Point(14) = {20e-3-radius, -H/2+20e-3, 0, h};
Point(15) = {20e-3, -H/2+20e-3+radius, 0, h};
Point(16) = {20e-3, -H/2+20e-3-radius, 0, h};

Circle(11) = {15, 12, 14};
Circle(12) = {14, 12, 16};
Circle(13) = {16, 12, 13};
Circle(14) = {13, 12, 15};
Line Loop(3) = {11:14};

//Hole 3
radius = 10e-3;
Point(17) = {L-28.5e-3, -H/2+51e-3, 0, h}; //Centre of the hole
Point(18) = {L-28.5e-3+radius, -H/2+51e-3, 0, h};
Point(19) = {L-28.5e-3-radius, -H/2+51e-3, 0, h};
Point(20) = {L-28.5e-3, -H/2+51e-3+radius, 0, h};
Point(21) = {L-28.5e-3, -H/2+51e-3-radius, 0, h};

Circle(15) = {20, 17, 19};
Circle(16) = {19, 17, 21};
Circle(17) = {21, 17, 18};
Circle(18) = {18, 17, 20};
Line Loop(4) = {15:18};

//Final Surface
Plane Surface(1) = {1, 2, 3, 4};

//Crack
Line{6} In Surface{1};

//Outputs
Physical Surface(2) = {1};
Physical Line(2) = {7,8,9,10};
Physical Line(3) = {11,12,13,14};