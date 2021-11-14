import util.Random
import java.util.Scanner
import java.util.InputMismatchException
import scala.collection.parallel.immutable.ParVector

object Main
{
  // Constantes del programa
  val columnas = 7
  val filas = 9
  val sc = new Scanner(System.in)
  
  // Generar un diamante aleatorio entre 1 y 6
  def diamanteAleatorio() = 1 + Random.nextInt(6)
  
  // Generar diamantes para los huecos vacíos (cuando es 0)  
  def generarDiamantes(x: Int): Int = x match
  {
    case 0 => diamanteAleatorio()
    case _ => x
  }
  
  // Función que imprime el tablero simulando las rejillas del juego con sus diamantes
  def imprimirTablero(l: List[Int]): Unit = l match
  {
    case Nil => println("-----------------------------")
    case a1::a2::a3::a4::a5::a6::a7::l => println("-----------------------------");
                                          print("| "); print(a1); print(" | "); print(a2); print(" | "); print(a3);
                                          print(" | "); print(a4); print(" | "); print(a5); print( " | "); print(a6);
                                          print(" | "); print(a7); println(" |"); imprimirTablero(l);
    case _ => throw new Error("Tablero erroneo")
  }
  
  // Función para mover un diamante desde una posición inicial a una final del tablero
  def mover(posInicial: Int , posFinal: Int, l: List[Int]): List[Int] =
    l.par.updated(posFinal - 1, l.par.apply(posInicial - 1)).updated(posInicial - 1, l.par.apply(posFinal - 1)).toList

  // Función que lee una fila de una lista pasando su posición
  def fila(pos: Int, l: List[Int]): List[Int] =
    l.par.drop((pos - 1) * columnas).take(columnas).toList

  // Función que lee una columna de una lista pasando su posición
  def columna(pos:Int, m: List[Int], l: List[Int] = Nil): List[Int] = m match
  {
  	case Nil => l
  	case _ => columna(pos, m.par.drop(columnas).toList, l ::: List(fila(1, m).par.apply(pos - 1)))
  }  
  
  // Convierte una fila y una columna a una posición de la lista
  def posicion(fila: Int, columna: Int): Int = ((fila - 1) * columnas) + columna
  
  // Comprueba si una posición dada (en fila y columna) está dentro del rango válido
  def esPosicionValida(fila: Int, columna: Int): Boolean = 
    fila >= 1 && columna >= 1 && fila <= filas && columna <= columnas
   
  // Comprueba si un movimiento dado (arriba, abajo, izquierda o derecha) es válido y devuelve la posición calculada  
  def esMovValido(dir: String, pos: Int): Int = dir match
  {
    case "u" if pos > columnas => pos - columnas 
    case "d" if pos <= (filas * columnas) - columnas => pos + columnas
    case "l" if pos % columnas != 1 => pos - 1 // Columna 1 
    case "r" if pos % columnas != 0 => pos + 1 // Última columna
    case _ => -1
  }
  
  // Elimina del tablero cuando se producen alineamientos de tres diamantes en horizontal sustituyendo dichos diamantes por 0
  def eliminar(l: List[Int], pos: Int = 2): List[Int] = pos match
  {
    case x if x == columnas * filas => l
    case x if x % columnas == 0 => eliminar(l, x + 2)
    case x if (l.par.apply(x - 2) == l.par.apply(x - 1) && l.par.apply(x - 1) == l.par.apply(x)) => eliminar(l.par.updated(x - 2, 0).updated(x - 1, 0).updated(x, 0).toList, pos + 1)
    case _ => eliminar(l, pos + 1)
  }
  
  // Agrega ceros al principio del tablero para luego sustituirlos por diamantes aleatorios
  def agregarCeros(l: List[Int], ceros: Int, pos: Int = 1) : List[Int] = ceros match
  {
    case 0 => l
    case _ => agregarCeros(l.par.updated(pos - 1, 0).toList, ceros - 1, pos + 1)
  }
  
  // Baja los diamantes de una columna 
  def bajar(col: List[Int], l: List[Int], pos: Int) : List[Int] = col match
  {
    case Nil => l
    case _ if col.head == 0 => bajar(col.tail, l, pos)
    case _ => bajar(col.tail, l.par.updated(pos - 1, col.head).toList, pos + 1)
  }
  
  // Se coloca la nueva columna en el tablero
  def colocarCol(l: List[Int], col: List[Int], numCol: Int, pos: Int = 1) : List[Int] = col match
  {
    case Nil => l
    case _ => colocarCol(l.par.updated(posicion(pos, numCol) - 1, col.head).toList, col.tail, numCol, pos + 1)
  }
  
  // Recoloca el tablero, para ello agrega los ceros necesarios, baja los diamantes, genera nuevos diamantes y vuelve a colocar la columna en el tablero
  def recolocar(l: List[Int], numCol: Int = columnas) : List[Int] = numCol match 
  {
    case 0 => l
    case _ => {
      val col = columna(numCol, l)
      val ceros = col.par.count(_ == 0)   
      recolocar(colocarCol(l, bajar(col, agregarCeros(col, ceros), ceros + 1).par.map(generarDiamantes).toList, numCol), numCol - 1)
    }    
  }
  
  // Se le pide al usuario que entre una fila y columna empezando por 1
  def elegirPosicion(): Int = 
  {
    try
    {
      print("Fila (1-" + filas + "): ")
      val fila = sc.nextInt()
      
      print("Columna (1-" + columnas + "): ")
      val colum = sc.nextInt()
      
      val pos = posicion(fila, colum)
      
      if (esPosicionValida(fila, colum))
      {
        pos
      }
      else
      {
        println("Elige una posición válida. \n") 
        elegirPosicion() // Cola recursiva y evitamos stackoverflow por parte del usuario
      }
    }
    catch
    {
      case e: InputMismatchException =>
      {
        println("Elige una posición válida. \n")
        sc.nextLine()
        elegirPosicion()
      }
    }
  }
  
  // Se pide al usuario que elija un movimiento (arriba, abajo, izquierda o derecha)
  def elegirMovimiento(posicion: Int): Int =
  {
    print("Movimiento (up,down,left,right) = (u,d,l,r): ")
    
    val posValida = esMovValido(sc.next(), posicion)
    
    if (posValida != -1)
    {
      posValida
    }
    else
    {
      println("Movimiento inválido")
      elegirMovimiento(posicion) // Cola recursiva y evitamos stackoverflow por parte del usuario
    }
  }
  
  // Muestra el tablero, elimina los alineamientos de tres, y si es necesario vuelve a eliminar y recolocar tantas veces como posibles alineamientos se encuentre en el nuevo tablero
  def calcularTablero(tablero: List[Int]): List[Int] = 
  {
    imprimirTablero(tablero)
    
    val eliminado = eliminar(tablero)
    
    if (eliminado.par.count(_ == 0) == 0) 
    {
      eliminado
    } 
    else
    {
      println("ELIMINAR")
      imprimirTablero(eliminado) 
      println("RECOLOCAR")
      calcularTablero(recolocar(eliminado))  
    }
  }
  
  // Se juega el movimiento que el usuario ha solicitado
  def jugar(tablero: List[Int]): List[Int] =
  {      
      val posInicial = elegirPosicion()
      
      calcularTablero(mover(posInicial, elegirMovimiento(posInicial), tablero))
  }
   
  // Bucle del juego que se ejecuta con una cola recursiva
  def juego(tablero: List[Int]): Unit =
  {
      print("Elige Mover (m) o Salir (s): ")
 
      sc.next() match
      {
        case "s" => println("Gracias por jugar. Un saludo.")
        case "m" => juego(jugar(tablero))
        case _ => juego(tablero) // Cola recursiva y evitamos stackoverflow durante toda la duración del juego
      }
  }
  
  // El método principal que crea el tablero y llama al bucle del juego. También es posible hacer extends de App y evitar este método
  def main(args: Array[String]): Unit =
  {
     juego(calcularTablero(ParVector.fill(filas * columnas)(diamanteAleatorio).toList))
  }    
}