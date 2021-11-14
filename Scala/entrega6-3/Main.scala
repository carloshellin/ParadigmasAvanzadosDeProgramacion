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
  def recolocar(l: List[Int], f: Int => Int = (x: Int) => x, numCol: Int = columnas) : List[Int] = numCol match 
  {
    case 0 => l
    case _ => {
      val col = columna(numCol, l)
      val ceros = col.par.count(_ == 0)   
      recolocar(colocarCol(l, bajar(col, agregarCeros(col, ceros), ceros + 1).par.map(f).toList, numCol), f, numCol - 1)
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
      calcularTablero(recolocar(eliminado, generarDiamantes))
    }
  }
  
  // Cuenta la cantidad de tríos que se encuentran en un tablero
  def contarAlineamientos(l: List[Int], pos: Int = 2, cont: Int = 0): Int = pos match
  {
    case x if x == columnas * filas => cont
    case x if x % columnas == 0 => contarAlineamientos(l, x + 2, cont)
    case x if l.par.apply(x) == 0 => contarAlineamientos(l, x + 1, cont)
    case x if (l.par.apply(x - 2) == l.par.apply(x - 1) && l.par.apply(x - 1) == l.par.apply(x)) => contarAlineamientos(l.par.updated(x - 2, 0).updated(x - 1, 0).updated(x, 0).toList, pos + 1, cont + 1)
    case _ => contarAlineamientos(l, pos + 1, cont)
  }
  
  // Devuelve el mejor movimiento válido de un tablero
  def mejorMovValido(l: List[Int], dir: String, actual: Int = 0, mejor: (Int, Int) = (-1, -1),  pos: Int = 1): (Int, (Int, Int)) = pos match
  {
    case x if x == columnas * filas => (actual, mejor)
    case _ => 
    {
      esMovValido(dir, pos) match
      {
        case -1 => mejorMovValido(l, dir, actual, mejor, pos + 1)
        case x =>
        {
          val movimiento = mover(pos, x, l)
          val contador = contarAlineamientos(movimiento)
                
          if (contador > 0 && contador >= actual)
          {
            mejorMovValido(l, dir, contador, (pos, x), pos + 1)
          }
          else
          {
            mejorMovValido(l, dir, actual, mejor, pos + 1)
          }
        }
      }
    }
  }
  
  // El ordenador busca el mejor movimiento
  def ordenador(l: List[Int]): (Int, Int) =
  {
    val (actualR, mejorR) = mejorMovValido(l, "r") // Mejor movimiento a la derecha
    val (actual, mejor) = mejorMovValido(l, "d", actualR, mejorR) // Mejor movimiento entre abajo y derecha
    mejor
  }
   
  // Elige aleatoriamente una dirección en base a una posición
  def dirAleatoria(pos: Int): Int = 
    esMovValido(Random.shuffle(List("u", "d", "l", "r")).head, pos) match
    {
      case -1 => dirAleatoria(pos)
      case x => x
    }
  
  // Bucle del juego que se ejecuta con una cola recursiva
  def juego(tablero: List[Int]): Unit =
  {
      val (posInicial, posFinal)  = ordenador(tablero)
      
      if ((posInicial, posFinal) == (-1, -1))
        println("No hay eliminaciones posibles. Se elegirá un movimiento aleatorio.")
      else
        println("Mejor movimiento del ordenador: " + posInicial + " a " + posFinal)
      
      print("Elige Mover (m) para usar el mejor movimiento del ordenador o Salir (s): ")
 
      sc.next() match
      {
        case "s" => println("Gracias por jugar. Un saludo.")
        case "m" if ((posInicial, posFinal) == (-1, -1)) =>
          {
            val posInicialAleatoria = Random.nextInt(filas * columnas)
            val posFinalAleatoria = dirAleatoria(posInicialAleatoria)
            println("Movimiento realizado: " + posInicialAleatoria + " a " + posFinalAleatoria)
            juego(calcularTablero(mover(posInicialAleatoria, posFinalAleatoria, tablero)))
          }
        case "m" => juego(calcularTablero(mover(posInicial, posFinal, tablero)))
        case _ => juego(tablero) // Cola recursiva y evitamos stackoverflow durante toda la duración del juego
      }
  }
  
  // El método principal que crea el tablero y llama al bucle del juego. También es posible hacer extends de App y evitar este método
  def main(args: Array[String]): Unit =
  {
    juego(calcularTablero(ParVector.fill(filas * columnas)(diamanteAleatorio).toList))
  }    
}