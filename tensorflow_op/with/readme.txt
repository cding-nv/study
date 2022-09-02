可以看到，整个运行过程如下：
（１）enter()方法被执行；
（２）enter()方法的返回值，在这个例子中是”Foo”，赋值给变量sample；
（３）执行代码块，打印sample变量的值为”Foo”；
（４）exit()方法被调用；

【注：】exit()方法中有３个参数， exc_type, exc_val, exc_tb，这些参数在异常处理中相当有用。
exc_type：　错误的类型
exc_val：　错误类型对应的值
exc_tb：　代码中错误发生的位置

https://blog.csdn.net/u012609509/article/details/72911564
