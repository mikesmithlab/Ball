def testfunction(var):
    print('success',var)



method_to_call = locals()['testfunction']('message')
