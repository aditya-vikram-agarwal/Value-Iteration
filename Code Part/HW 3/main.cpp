/*
This file includes all of the code for HW3. The lines that you must fill in are in one location, labeled
with "// TODO".

If you are new to C++, please follow some online tutorials to get a feel for C++. Don't worry about "pointers"
or "makefiles", as you won't need to know about them for these assignments.

Some pointers:
1. In C++ you can compile in "release" or "debug" mode In Visual Studio and CLion, you should see a drop-down box
near the top of the IDE that says "Debug" or "Release". Changing this changes the configuration. Debug runs slower
(still more than fast enough for this assignment), but allows you to place break-points in your code, step through
your code as it runs watching variable values change, and also provides more information if your code crashes. Release
results in code that runs *much* faster. However, it will give cryptic errors when it crashes (not telling you the 
line that it crashed on for example) and does not allow you to use most of the features of your debugger. Generally,
you use debug mode when soemthing isn't working right or when first running your code, and then switch to release.

2. In visual studio, hit F7 to compile your code. If there are errors, hit F4 to cycle through the errors. Run your code with F5.
If you are in release mode, run with ctrl+F5. Place a breakpoint with F9, step forwards with F10. Right click a breakpoint to set a
condition, like to only break when s=22. F11 will step into a function, while Shift+F11 will step out of the current function. Also, if you
select a variable or function in the editor and hit F12, it will take you to the definition of that function/variable. If you right click a
variable or function, you can click "peek definition" to bring up a small window showing you the definition of that term.

3. In CLion, build by navigating to "build" at the top, and selecting the option "build". Then run with "Run-> Debug 'Project Name'", where
this project's name is "CLion". If you are in release mode, use "Run->Run 'Project Name'". Under the run menu you can also find the commands
for adding breakpoints. 

4. I've added comments into the document below so that you can start to get an idea for how this code and C++ work. Start with the function "main"
near the bottom of this file.

5. This code uses "Eigen", a linear algebra library for C++. Once you've looked a bit at C++ in general below, check out the getting started page for Eigen: http://eigen.tuxfamily.org/dox/GettingStarted.html
Specifically, you'll be using VectorXd and MatrixXd objects. Notice that for a VectorXd v, you can reference the i'th element as v[i] or v(i). For a MatrixXd m, you
can reference the i,j'th element as m(i,j), but not m[i,j]. Also, notice that you can call m.row(i), which returns an object essentially like a VectorXd that corresponds to the i'th row.
You can call v.transpose() on a vector to transpose it. You can call v.dot(u) for two vectors v and u. You can call v.maxCoeff() to get the largest element of v, or v.minCoeff() to get the smallest.
Notice that with vector objects v, you must write v[i], not v(i) like you can write with VectorXd objects. Also, you cannot "combine" vector and VectorXd objects - they are entirely different beasts.
E.g., you can't write v.dot(u) if v and u are not both VectorXd objects. Think of vector objects as C++'s standard "array" object, which is not meant for linear algebra.

6. Once you're looked into the code below, notice that P is a vector, each element of which is a matrix. To get P(s,a,s'), you need to write P[s](a,s'). You can grab a whole row with P[s].row(a), for example.
*/

// These statements are saying to include source code that is stored somewhere else (these come with most compilers)
#include <iostream>		// For console i/o
#include <vector>		// For arrays that we don't use for linear algebra
#include <string>		// For strings used in file names
#include <fstream>		// For file i/o
#include <iomanip>		// For setprecision when printing numbers to the console or files
#include <algorithm>

// This statement includes a library for linear algebra. We included this in the "lib" folder, and have set up your project
// to look in this folder for the library.
#include <Eigen/Dense>	// For vectors and matrices that we use for linear algebra

using namespace std;			// Some terms below are "inside" std. For example, cout, cin, and endl. Normally you have to write std::cout. This line makes it so that you don't have to write std:: before everything you are using from standard libraries.
using namespace Eigen;			// Like the above line, but for the Eigen library that we are using for linear algebra. Instead of Eigen::VectorXd, you can just write VectorXd with this.

/*
This class represents an MDP with a finite number of states and actions.
*/
class MDP
{
public:	// You can declare functions and variables to be "private", meaning that you can only reference them inside of the class. This "public" means that anywhere else that you have an MDP object, you can reference these functions and variables.

	// Member variables of the MDP class.
	int numStates;					// States are integers in the range 0 to numStates-1
	int numActions;					// Actions are integers in the range 0 to numActions-1
	vector<MatrixXd> P;		// [numStates][numActions][numStates]. We write vector<Foo> to create a vector (C++'s array) of Foo objects. So, this is an array of "MatrixXd" objects. It will have length "numStates", and the matrices it holds will have numActions rows and numStates columns.
	MatrixXd R;				// [numStates][numActions]

	double gamma;					// "double" means "floating point, typically 64-bit. "float" means "floating point, typically 32-bit".

	// An MDP object has this function, which loads an MDP from the file with the specified file name. Here "void" means this function doesn't return anything, const means that "fileName" cannot be changed within this function, & means that when
	// you call this function you don't pass a copy of the string "fileName", you pass the actual object "fileName", so any changes to it within the function would change it outside the function. However, "const" ensures that it won't be changed.
	void loadFromFile(const string& fileName)
	{
		// ifstream = input file stream. This is the object for reading from files.
		ifstream in(fileName.c_str());	// This creates an ifstream object and calls the "constructor" for the object - a function called when the object is created with the argument fileName.c_str(). The constructor for ifstream is designed to open the provided fileName. The constructor expects a char* (a default C++ string type). They are a pain to work with, so we are using "string" types. The .c_str() says to convert the string to a char* before passing it.
		in >> numStates >> numActions;
		cout << "numStates = " << numStates << endl;

		// Resize P to be an std::vector containing numStates matrices, each of which has numActions rows and numStates cols. So, P[s](a,sPrime) is the probability of transitioning to sPrime from s when taking action a.
		P.resize(numStates);	// Make P a vector of length numStates.
		// If we do not include brackets, for, while, if, else, etc. statements implicitly have brackets around the next line ONLY. So, the for loop below only includes the P[s] line. C++ doesn't care about white space, so tabs and spaces
		// are just to help us see what is going on.
		for (int s = 0; s < numStates; s++)
			P[s].resize(numActions, numStates);

		// Read in the transition probabilities. Notice that the outer loop is over actions. This loop order makes it easier to manually enter transition probabilities for gridworlds like 687Gridworld.txt
		for (int a = 0; a < numActions; a++) // This for-loop only includes the next "line", which is really the next command
			for (int s = 0; s < numStates; s++)	// which is this for loop, which only includes the next line
				for (int sPrime = 0; sPrime < numStates; sPrime++) // which is this for loop. So, these loops are all nested. The next for loops below use brackets to show an equivalent way of writing this (that takes more space)
					in >> P[s](a, sPrime);

		// Load the reward function
		R.resize(numStates, numActions);
		for (int s = 0; s < numStates; s++)
		{
			for (int a = 0; a < numActions; a++)
			{
				in >> R(s, a);
			}
		}

		// Get the reward discount parameter.
		in >> gamma;

		// Close the input file - we are done reading from it.
		in.close();
	}

	// Run a sanity check on a loaded MDP - can find some errors in a loaded MDP file.
	bool sanityCheck()
	{
		// Make sure the different parameters take reasonable values
		if ((numStates <= 0) || (numActions <= 0) || (gamma > 1) || (gamma < 0))
			return false;
		// Make sure that P(s,a,*) is a probability distribution
		for (int s = 0; s < numStates; s++)
		{
			// Entries must all be in [0,1]
			if ((P[s].maxCoeff() > 1) || (P[s].minCoeff() < 0))
				return false;
			// Sum over next-state probabilities must be one (with some error for floating point issues)
			for (int a = 0; a < numActions; a++)
				if (fabs(P[s].row(a).sum() - 1.0) > 0.0000001)
					return false;
		}
		// All tests passed
		return true;
	}
};

/*
Run value iteration on the provided MDP. The output is the optimal value function.
M: MDP to run value iteration on
print: If set to true, valueIteration prints the sequence of value function approximations it produces.
epsilon: Tolerance parameter. We will terminate when every element of v_{k+1} is within epsilon of the corresponding element in v_{k}

The "=" value after epsilon is a default value - if you do not provide the epsilon parameter it will be set to this value automatically. If you do provide this argument to the function, this value will be replaced with the value you provide.
*/
VectorXd valueIteration(const MDP& M, const bool & print = false, const double& epsilon = 0.000000001)
{
	// Create arrays for our current and new value function estimates, initialized to zero
	VectorXd vCur = VectorXd::Zero(M.numStates), vNew(M.numStates);	// VectorXd::Zero(n) is a vector of all zeros, of length n.

	// Create array used when computing the max_a in the Bellman operator
	VectorXd temp(M.numActions);	// Create a vector, filled with unknown values for now, but of length M.numActions.
	// Notice above "M.numActions" is how we reference the variable "numActions" in the MDP class we defined above. Similarly, M.sanityCheck() runes the sanityCheck function that we defined.

	// Iteration loop:
	for (int itCount = 0; true; itCount++)
	{
		// If the "print" argument is true, print out extra information.
		if (print)
		{
			cout << "Value function estimate number " << itCount << ":" << endl << vCur.transpose() << endl << "Press enter to continue." << endl;
			cin.ignore(cin.rdbuf()->in_avail()); getchar();	// Wait until user presses enter. If you're new to C++, don't worry about this line and why it is so complicated...
		}

		// For each state, compute vNew[s] given vCur.
		for (int s = 0; s < M.numStates; s++)
		{
			// TODO: You must enter the code here. It should load vNew[s] (the value next value function estimate for state s) based on vCur (the previous value function estimate)
			// and the information stored in the MDP object M. Hint: read the rest of this code and understand it. This assignment can be completed by inserting one line
			// below (mine extends to column 76, including whitespace). The missing line resembles another line somewhere in this code.
			
			
			for (int a = 0; a < M.numActions; a++) {
				temp[a] = 0;
				for (int s_dash = 0; s_dash < M.numStates; s_dash++) {
					temp[a] += M.P[s](a, s_dash) * (M.R(s, a) + M.gamma * vCur[s_dash]);
				}
			}
				
			vNew[s] = temp.maxCoeff();

		}

		// Check for termination. Take element-wise absolute value of (vNew-vCur). If the max element is <= epsilon, we are done.
		// Note: Eigen provides the abs() function for array objects, not VectorXd objects. We can view the VectorXd as an array though, with the .array() function.
		if ((vNew - vCur).array().abs().maxCoeff() <= epsilon)
		{
			cout << "Value iteration finished in " << itCount + 1 << " iterations." << endl;
			break; // Break out of the while(true) loop
		}

		// Move vNew into vCur in preparation for next iteration of the while loop
		vCur = vNew;
	}

	// Return the latest value function
	return vNew;
}

// This function prints the optimal value function and optimal policies to a file. Epsilon is a tolerance parameter when checking if two
// actions are roughly equally optimal.
void print(const Eigen::VectorXd& vStar, const MDP& M, const string& fileName, const double & epsilon = 0.000000001)
{
	ofstream out(fileName);	// Open the file for printing

	// Print vStar
	out << "Optimal value function:" << endl;
	out << setprecision(10) << vStar.transpose() << endl << endl; // Here "setprecision" comes from the <iomanip> include, and says how many decimals to include when printing.

	// Compute and print the optimal policy for each state
	VectorXd temp(M.numActions);
	out << "Optimal policies:" << endl;
	for (int s = 0; s < M.numStates; s++)
	{
		// From M, get the s'th row of the reward function, transpose it to be a column vector, add to it gamma (a floating poitn number) times the transition matrix for state s (which is a matrix with numActions rows and numStates cols) times the optmial value function (a column vector of length numStates)
		temp = M.R.row(s).transpose() + M.gamma * M.P[s] * vStar;	// Compute the expected return if each action is taken from this state and an optimal policy is followed thereafter
		double maxActionValue = temp.maxCoeff();					// Get the largest expected return over all possible actions.
		// Get all actions that achieve the maxActionValue
		vector<int> optimalActions(0);	// vector<int> is creating a vector object that holds "int" objects. The (0) here says initialize to length zero.
		for (int a = 0; a < M.numActions; a++)
			if (fabs(temp[a] -maxActionValue) < epsilon)
				optimalActions.push_back(a);						// You can treat an std::vector like a stack, pushing and popping. 
		// Print all actions that are optimal in this state
		out << "[";
		for (int i = 0; i < (int)optimalActions.size() - 1; i++)	// optimalActions.size() is an unsigned int. We write "(int)" to cast this to a regular integer so that we can compare it to i. An alternative is to make i and "unsigned int"
			out << optimalActions[i] << ",";
		out << optimalActions[optimalActions.size() - 1] << "]\t";	// We handle the last action separately to not print a comma after it			
	}

	// Close the output file.
	out.close();
}

// Given a filename, load the file into an MDP, run sanity checks on this MDP, run value iteration, and print the result to a file.
void run(const string& fileName)
{
	MDP M;				// Where we will store the loaded MDP. This object is defined above in "class MDP".
	VectorXd vStar;		// Where we will store the optimal value function. This object is from the Eigen linear algebra library.

	cout << "Loading " << fileName << " into MDP object..." << endl;
	M.loadFromFile("../../../input/" + fileName);					// In our MDP class we have a loadFromFile function that is called here. "input/" is a char*, the c++ built-in string. fileName is a standard library string. The + operator appends these and returns a standard library string. When new to C++, try to stick with standard library strings rather than using char*, except when manually entering a string in your code like here.

	cout << "Loading complete. Running sanity checks." << endl;
	if (M.sanityCheck())	// Without {  }, if statements, while loops, for loops, etc. are only over the following line. If you want more than two subsequent lines to be in the if statement (while loop, for loop, else if statement, etc) then use {  } as in the if-statement in main.
		cout << "Sanity checks on MDP passed." << endl;
	else
	{
		cout << "Sanity checks on MDP failed!" << endl;
		cout << "Aborting this run." << endl;
		return;
	}
	
	cout << "Running value iteration..." << endl;
	// Comment out the upper line and uncomment the lower line to have value iteration print the value function estimates and wait for you to hit 'enter' after every iteration (can help when debugging).
	vStar = valueIteration(M);				// Run the value interation function that you will be writing. It is defined in valueIteration.hpp and implemented in valueIteration.cpp
	//vStar = valueIteration(M, true);				// Run the value interation function that you will be writing. It is defined in valueIteration.hpp and implemented in valueIteration.cpp

	cout << "Printing results to file..." << endl;
	print(vStar, M, "../../../output/" + fileName);	// Prints the optimal value function and optimal policy to the specified file
	cout << "Run on " << fileName << " complete." << endl << endl;
}

// Entry point for the C++ program - it begins execution here.
// argc will be the number of command line arguments + 1, and argv will be the path to the executable followed by
// all of the command line arguments. To complete this assignment you do not need to know about or use command line arguments.
// These are what *we* will use when testing your program.
//
// To provide command line arguments in visual studio (so you don't have to type the file name every time), click
// Project -> ValueIteration Properties -> Configuration Properties -> Debugging -> Command arguments.
// Enter into this box the arguments that you would like to pass, like:
// 687Gridworld.txt myTestMDP.txt
//
// Be sure in the properties box that the top left "Configuration" matches the configuration you are using (or do this for "all configurations", both release and debug).
int main(int argc, char* argv[])
{
    if (argc == 1)		// Check if there were no command line arguments (the first is the path to the running executable)
	{
		// Ask the user for a file name and run value iteration on it
		string fileName;		// This object is from the standard library and can store strings. It's a nice alternative to char*, the default string type in C++.
		cout << "Enter file name (within input directory): ";	// "cout" means "console out" -- print to the console. The "<<" is used to separate terms that should be printed. See some of the other cout statements above for examples.
		cin >> fileName;										// "console in" -- get input from the user.
		run(fileName);	// Above we created a function "run". This calls it. Notice that in C++ you can only call functions defined above your current location in the code. (You can get around this by "defining" a function in one place and then implementing it later - you only need it to be defined before it can be called. For now, if you are new to C++, just ensure that functions you want to call are above where you call them).
	}
	// else if -- if you want an "else if" block, it would be like this, as two words.
	else
	{
		// Loop over the command line arguments, running value iteration on all of the listed files. Start with i=1 since argv[0] is the path to the running executable
		for (int i = 1; i < argc; i++)	// This is the format for a for-loop in C++. The first part declares an integer object i, and initializes it to one. The middle bit is the condition that, when satisfied, allows the loop to continue. The final part, i++, is shorthand for i=i+1.
			run(argv[i]);	// Call our run function.
	}
	cout << "Done." << endl;
}