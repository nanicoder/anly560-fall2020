package ControlStructures;

public class ControlStuctures {

	public static void main(String[] args) {
		//if statement
		int x = 10;
		
		if(x <20) {
			System.out.println("This is true part");
		}else {
			System.out.println("This is false part");
		}
		
		
		//switch statement
		
		int day = 10;
		String dayString = null;
		switch (day) {
		case 1: 
			dayString = "Monday";
			break;
		case 2:
			dayString = "Tuesday";
			break;
		case 3:
			dayString = "Wednesday";
			break;
		case 4:
			dayString = "Thurdsday";
			break;
		case 5:
			dayString = "Friday";
			break;
		case 6:
			dayString = "Saturday";
			break;
		case 7:
			dayString = "Sunday";
			break;
		default:
			dayString = "Invalid day";
			break;
		}
		System.out.println(dayString);
		
		
		//for loop
		for (int z =0; z <= 20; z = z+1) {
			System.out.println("Value of z in teh loop : "+ z);
		}
		
		//while loop
		int y = 5;
		while(y<10) {
			System.out.print("Value of y: "+ y);
			y++;
			System.out.print("\n");
		}
		
		//do while loop
		do{
			System.out.print("Value of x: "+ x);
			x++;
			System.out.print("\n");
		}while (x<20);
	}

}
