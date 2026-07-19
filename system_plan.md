This contain the plan fo restrucutreing the system realted tempalte and how its organised
architectures:
root optimizer lib
    -system
        -system class<- this is abstract class where all importna api calls are predefined and all utitlity py files are improted
        -utility pythong files are deifned here and importn to the class
for a new project the user will make a simlar strucutred "system.py" that import this class and build absed on this(for an example you can condider how cosutom rl env is made eg: Env=gym.Env,class new_env(Env) or simlar the idea is the api calls are implement without any fail and all such calls will have a predefined input and output rest is modular)

The system is defined based on parameter dict "params":
-within this params there will be divided in to two parts:
    --primary:this defines the system,eg: N,dt,coupling params etc
    --secondary:these are the type or parasm that can be tuned or changed wihtin a optimisation like weights etc
a rough idea of what params looks like and how it will be used is :
system_new=system(p_params,s_params)
type(p_params,s_params)={}
p_params will have(mandatory ones will be):
-T(represent the final time,can be alsu called as tau)
-N(total numbe of points in the total stats)
-dt(the time step)
-control_dim(passed as array)[n,N]
-control_channels(just the names like ux uy,uz in the cornetc oder of rows,this will help to use and reacll certain contrl).like contrls["ux"] means ux chnnael-we need to verify this with optimizer lib that it rakes row wise(else we have to choos column wise),ie the controls are passed wth the info on their namies/ids in tin intialisiation,but later controls are address as a array thats vectorised or so to work easier
-state_dim:this just tells the system,lib about the expeted dimentiosn  it will be psi of 2 or more lvenl or rho ie a array [m1,m2]
-see if there is anythign more
s_parms:
    -mostle weights of the difffert pbjexctive that are used to tune while doing cariculum training or so

The usual work that is encounterd we call this as "open_loop_gradient" type(for now we only have this kind of system so we deisng based on this but lib is genaral)

more info about "open_loop_gradient" type situations:
These are based on some analytical thoer were on could derive the optimsiation aprametred,dynamics analytically.
some common methods:
->forward propagation-forward_prop(controls):
    -this intake the control then do forward propgation of the physcial system,which result in the states,trajectory,bloachpshere coordinates,projection on x,y,z axis etc
->backward propogation-back_prop(control):
    -this intake the control use the vaibal for system class(like last state ,or somthing) then do the backward propogation and get all costates.
    -usually backward is allowed only after forward sicne it depend on som info from the forward
->gradient(control):the idea od this functionis to do forward,backward then find the gradiednt to udpade the contrls and pass it.we can define and edit this so that it can pass the griadent and some other relvent info if eneded
->simulate(contorl)-optional:this measn for situation were we have to evlaute over enspeble this will be done via this function
->evlaute(control): this do the relenvent propogation anf give the reuslts back as a dict or array(we need to deing this part so that it is easily handeld by the lib),this can pass the states costaes etc as well so if one need to plot bloch he can do it the diea is we can deinf indicualt functiona nd fucntion that loads all at asme time but the.  importnat factor is the efficeny we dont want to recomupte every time
->cache_reset():removes all aprams and clear the memeory
->describe():the idea is this give the params value and importn info progmrmilaccal for quik checks
->metrics_build(control):this passes all the metrics as a dict, for a given control,note this means it has to do forward,backward and some evlaution as well so inorder to make these efficent we need to make prorpe strucutre
->metrics():this just show whats updated by rpesious control,the metrics can be updated when evlaute,simulate are used previously .we can add ore such efficent way so its not a misleading but efficent one

->mention if we need any more function that very importnant like jacobian double derivate wrt u etc
Note: the idea is to have a set of predefined function that help the lib to do api call sporply but not to overengieneer and make it too specific ,we need a geric type for the mentioend type
