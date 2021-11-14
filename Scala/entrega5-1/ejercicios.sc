object ejercicios {
	// Genera una lista de 1 a n enteros
  def iota(n: Int): List[Int] =
  	if (n == 0) Nil else iota(n - 1) ::: List(n)
                                                  //> iota: (n: Int)List[Int]
  // Suma los elementos de una lista
  def suma(l: List[Int]): Int = l match
  {
  	case Nil => 0
  	case cabeza :: resto => cabeza + suma(resto)
  }                                               //> suma: (l: List[Int])Int
  
  // Calcula el cuadrado de un número
  def cuadrado(x: Int): Int = x * x               //> cuadrado: (x: Int)Int
  // Suma los cuadrados de todos los enteros entre a y b
  def sumarCuadrados(a: Int, b: Int): Int =
  	if (a > b) 0 else cuadrado(a) + sumarCuadrados(a + 1, b)
                                                  //> sumarCuadrados: (a: Int, b: Int)Int
	// sumf1n realiza el sumatorio entre a y b de una función
  def sumf1n(f: Int => Int, a: Int, b: Int): Int =
  	if (a > b) 0 else f(a) + sumf1n(f, a + 1, b)
                                                  //> sumf1n: (f: Int => Int, a: Int, b: Int)Int
  // Función que comprueba si un número termina en 2 o 3 para usarlo con sumf1n
  def suma2o3(x: Int): Int =
  	if (x % 10 == 2 || x % 10 == 3) x else 0  //> suma2o3: (x: Int)Int

	// Cuenta las cifras de un número
  def contarCifras(x: Int): Int =
  	if (x / 10 == 0) 1 else 1 + contarCifras(x / 10)
                                                  //> contarCifras: (x: Int)Int
  // Función que se le pasa una lista y devuelve una nueva lista con los n primeros
  def tomar(n: Int, l: List[Int]): List[Int] =
  	if (n == 0 || l.isEmpty) Nil else l.head :: tomar(n - 1, l.tail)
                                                  //> tomar: (n: Int, l: List[Int])List[Int]
  // Función que se le pasa una lista y devuelve una nueva lista sin los n primeros
  def dejar(n: Int, l: List[Int]): List[Int] =
  	if (n == 0 || l.isEmpty) l else dejar(n - 1, l.tail)
                                                  //> dejar: (n: Int, l: List[Int])List[Int]
	// Función que imprime una lista como una matriz (por defecto de 4x4)
  def imprimir(l: List[Int], n: Int = 4): Unit = l match
  {
  	case Nil => return
  	case default => println(tomar(n, l)); imprimir(dejar(n, l), n)
  }                                               //> imprimir: (l: List[Int], n: Int)Unit
  
  // Función que lee un elemento de una lista pasando el índice
  def elemento(m: List[Int], indice: Int): Int =
		tomar(1, dejar(indice, m)).head   //> elemento: (m: List[Int], indice: Int)Int
  
  // Función que lee una fila de una lista pasando su posición (por defecto matriz de 8x8)
  def fila(m: List[Int], posicion: Int, n: Int = 8): List[Int] =
  	tomar(n, dejar(posicion * n, m))          //> fila: (m: List[Int], posicion: Int, n: Int)List[Int]
  
  // Función que lee una ccolumna de una lista pasando su posición (por defecto matriz de 8x8)
  def columna(m: List[Int], posicion:Int, n: Int = 8, l: List[Int] = Nil): List[Int] = m match
  {
  	case Nil => l
  	case default => columna(dejar(n, m), posicion, n, l ::: List(elemento(fila(m, 0), posicion)))
  }                                               //> columna: (m: List[Int], posicion: Int, n: Int, l: List[Int])List[Int]
  
  // Función que realiza la traspuesta (por defecto matriz de 8x8)
  def traspuesta(m: List[Int], n: Int = 8, l: List[Int] = Nil): List[Int] = n match
  {
  	case 0 => l
  	case default => traspuesta(m, n - 1, columna(m, n - 1) ::: l)
  }                                               //> traspuesta: (m: List[Int], n: Int, l: List[Int])List[Int]
  	
  iota(10)                                        //> res0: List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
  suma(iota(10))                                  //> res1: Int = 55
  sumarCuadrados(2, 3)                            //> res2: Int = 13
  sumf1n(cuadrado, 2, 3)                          //> res3: Int = 13
  suma2o3(12)                                     //> res4: Int = 12
  sumf1n(suma2o3, 0, 20)                          //> res5: Int = 30
  contarCifras(20)                                //> res6: Int = 2
  tomar(3, iota(10))                              //> res7: List[Int] = List(1, 2, 3)
  dejar(3, iota(10))                              //> res8: List[Int] = List(4, 5, 6, 7, 8, 9, 10)
  imprimir(List(0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5)) //> List(0, 1, 2, 3)
                                                  //| List(4, 5, 6, 7)
                                                  //| List(8, 9, 0, 1)
                                                  //| List(2, 3, 4, 5)
	imprimir(iota(64), 8)                     //> List(1, 2, 3, 4, 5, 6, 7, 8)
                                                  //| List(9, 10, 11, 12, 13, 14, 15, 16)
                                                  //| List(17, 18, 19, 20, 21, 22, 23, 24)
                                                  //| List(25, 26, 27, 28, 29, 30, 31, 32)
                                                  //| List(33, 34, 35, 36, 37, 38, 39, 40)
                                                  //| List(41, 42, 43, 44, 45, 46, 47, 48)
                                                  //| List(49, 50, 51, 52, 53, 54, 55, 56)
                                                  //| List(57, 58, 59, 60, 61, 62, 63, 64)
	elemento(iota(64), 0)                     //> res9: Int = 1
	fila(iota(64), 0)                         //> res10: List[Int] = List(1, 2, 3, 4, 5, 6, 7, 8)
  columna(iota(64), 0)                            //> res11: List[Int] = List(1, 9, 17, 25, 33, 41, 49, 57)
  traspuesta(iota(64))                            //> res12: List[Int] = List(1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42
                                                  //| , 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 
                                                  //| 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 3
                                                  //| 9, 47, 55, 63, 8, 16, 24, 32, 40, 48, 56, 64)
}