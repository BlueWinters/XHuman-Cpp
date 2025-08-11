
#ifndef __Face_Information__
#define __Face_Information__

#include <vector>

#define FaceAlignNumPoints		68
#define FaceIdentityInvalid		-1
#define ObjectFrequencyMax	  1000


#ifndef StdMax
#define StdMax(a,b)  (((a) > (b)) ? (a) : (b))
#endif
#ifndef StdMin
#define StdMin(a,b)  (((a) < (b)) ? (a) : (b))
#endif



class FaceObject
{
public:
	// base
	float score;
	int box[4];     // lft,top,rig,bot
	int points[10]; // le,re,ns,mt
	// align
	int landmarks[FaceAlignNumPoints*2];
	// for tracking
	int identity;
	int frequency;

public:
	FaceObject() : identity(-1), score(0.f), frequency(1),
		box(), points(), landmarks(){};

public:
	inline int hit()
	{
		frequency++; 
		return StdMin(frequency, ObjectFrequencyMax);
	}
};

typedef std::vector<FaceObject*> FaceObjectVector;

#endif

