namespace PHiLiP {
// See
// https://stackoverflow.com/questions/7395113/is-recursive-explicit-template-instantiation-possible

namespace Tests {
template <
    template <int, int> class ClassType,
    int dim,
    int nstate >
struct Instantiator
{
    ClassType<1, nstate> t1;
    ClassType<2, nstate> t2;
    ClassType<3, nstate> t3;
    Instantiator<ClassType, 1, nstate-1> i4;
    Instantiator<ClassType, 2, nstate-1> i5;
    Instantiator<ClassType, 3, nstate-1> i6;
};

template <template <int , int > class ClassType>
struct Instantiator<ClassType, 1, 1>
{
    ClassType<1,1> t;
};
template <template <int , int > class ClassType>
struct Instantiator<ClassType, 2, 1>
{
    ClassType<2,1> t;
};
template <template <int , int > class ClassType>
struct Instantiator<ClassType, 3, 1>
{
    ClassType<3,1> t;
};

}
}
